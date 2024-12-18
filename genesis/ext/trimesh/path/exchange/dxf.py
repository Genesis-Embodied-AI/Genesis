from collections import defaultdict

import numpy as np

from ... import grouping, resources, util
from ... import transformations as tf
from ...constants import log
from ...constants import tol_path as tol
from ...util import multi_dict
from ..arc import to_threepoint
from ..entities import Arc, BSpline, Line, Text

# unit codes
_DXF_UNITS = {
    1: "inches",
    2: "feet",
    3: "miles",
    4: "millimeters",
    5: "centimeters",
    6: "meters",
    7: "kilometers",
    8: "microinches",
    9: "mils",
    10: "yards",
    11: "angstroms",
    12: "nanometers",
    13: "microns",
    14: "decimeters",
    15: "decameters",
    16: "hectometers",
    17: "gigameters",
    18: "AU",
    19: "light years",
    20: "parsecs",
}
# backwards, for reference
_UNITS_TO_DXF = {v: k for k, v in _DXF_UNITS.items()}

# a string which we will replace spaces with temporarily
_SAFESPACE = "|<^>|"

# save metadata to a DXF Xrecord starting here
# Valid values are 1-369 (except 5 and 105)
XRECORD_METADATA = 134
# the sentinel string for trimesh metadata
# this should be seen at XRECORD_METADATA
XRECORD_SENTINEL = "TRIMESH_METADATA:"
# the maximum line length before we split lines
XRECORD_MAX_LINE = 200
# the maximum index of XRECORDS
XRECORD_MAX_INDEX = 368


def load_dxf(file_obj, **kwargs):
    """
    Load a DXF file to a dictionary containing vertices and
    entities.

    Parameters
    ----------
    file_obj: file or file- like object (has object.read method)

    Returns
    ----------
    result: dict, keys are  entities, vertices and metadata
    """

    # in a DXF file, lines come in pairs,
    # a group code then the next line is the value
    # we are removing all whitespace then splitting with the
    # splitlines function which uses the universal newline method
    raw = file_obj.read()
    # if we've been passed bytes
    if hasattr(raw, "decode"):
        # search for the sentinel string indicating binary DXF
        # do it by encoding sentinel to bytes and subset searching
        if raw[:22].find(b"AutoCAD Binary DXF") != -1:
            # no converter to ASCII DXF available
            raise ValueError("binary DXF not supported!")
        else:
            # we've been passed bytes that don't have the
            # header for binary DXF so try decoding as UTF-8
            raw = raw.decode("utf-8", errors="ignore")

    # remove trailing whitespace
    raw = str(raw).strip()
    # without any spaces and in upper case
    cleaned = raw.replace(" ", "").strip().upper()

    # blob with spaces and original case
    blob_raw = np.array(str.splitlines(raw)).reshape((-1, 2))
    # if this reshape fails, it means the DXF is malformed
    blob = np.array(str.splitlines(cleaned)).reshape((-1, 2))

    # get the section which contains the header in the DXF file
    endsec = np.nonzero(blob[:, 1] == "ENDSEC")[0]

    # store metadata
    metadata = {}

    # try reading the header, which may be malformed
    header_start = np.nonzero(blob[:, 1] == "HEADER")[0]
    if len(header_start) > 0:
        header_end = endsec[np.searchsorted(endsec, header_start[0])]
        header_blob = blob[header_start[0] : header_end]

        # store some properties from the DXF header
        metadata["DXF_HEADER"] = {}
        for key, group in [
            ("$ACADVER", "1"),
            ("$DIMSCALE", "40"),
            ("$DIMALT", "70"),
            ("$DIMALTF", "40"),
            ("$DIMUNIT", "70"),
            ("$INSUNITS", "70"),
            ("$LUNITS", "70"),
        ]:
            value = get_key(header_blob, key, group)
            if value is not None:
                metadata["DXF_HEADER"][key] = value

        # store unit data pulled from the header of the DXF
        # prefer LUNITS over INSUNITS
        # I couldn't find a table for LUNITS values but they
        # look like they are 0- indexed versions of
        # the INSUNITS keys, so for now offset the key value
        for offset, key in [(-1, "$LUNITS"), (0, "$INSUNITS")]:
            # get the key from the header blob
            units = get_key(header_blob, key, "70")
            # if it exists add the offset
            if units is None:
                continue
            metadata[key] = units
            units += offset
            # if the key is in our list of units store it
            if units in _DXF_UNITS:
                metadata["units"] = _DXF_UNITS[units]
        # warn on drawings with no units
        if "units" not in metadata:
            log.debug("DXF doesn't have units specified!")

    # get the section which contains entities in the DXF file
    entity_start = np.nonzero(blob[:, 1] == "ENTITIES")[0][0]
    entity_end = endsec[np.searchsorted(endsec, entity_start)]

    blocks = None
    check_entity = blob[entity_start:entity_end][:, 1]
    # only load blocks if an entity references them via an INSERT
    if "INSERT" in check_entity or "BLOCK" in check_entity:
        try:
            # which part of the raw file contains blocks
            block_start = np.nonzero(blob[:, 1] == "BLOCKS")[0][0]
            block_end = endsec[np.searchsorted(endsec, block_start)]

            blob_block = blob[block_start:block_end]
            blob_block_raw = blob_raw[block_start:block_end]
            block_infl = np.nonzero((blob_block == ["0", "BLOCK"]).all(axis=1))[0]

            # collect blocks by name
            blocks = {}
            for index in np.array_split(np.arange(len(blob_block)), block_infl):
                try:
                    v, e, name = convert_entities(blob_block[index], blob_block_raw[index], return_name=True)
                    if len(e) > 0:
                        blocks[name] = (v, e)
                except BaseException:
                    pass
        except BaseException:
            log.error("failed to parse blocks!", exc_info=True)

    # actually load referenced entities
    vertices, entities = convert_entities(
        blob[entity_start:entity_end], blob_raw[entity_start:entity_end], blocks=blocks
    )

    # return result as kwargs for trimesh.path.Path2D constructor
    result = {"vertices": vertices, "entities": entities, "metadata": metadata}

    return result


def convert_entities(blob, blob_raw=None, blocks=None, return_name=False):
    """
    Convert a chunk of entities into trimesh entities.

    Parameters
    ------------
    blob : (n, 2) str
      Blob of entities uppercased
    blob_raw : (n, 2) str
      Blob of entities not uppercased
    blocks : None or dict
      Blocks referenced by INSERT entities
    return_name : bool
      If True return the first '2' value

    Returns
    ----------
    """

    if blob_raw is None:
        blob_raw = blob

    def info(e):
        """
        Pull metadata based on group code, and return as a dict.
        """
        # which keys should we extract from the entity data
        # DXF group code : our metadata key
        get = {"8": "layer", "2": "name"}
        # replace group codes with names and only
        # take info from the entity dict if it is in cand
        renamed = {get[k]: util.make_sequence(v)[0] for k, v in e.items() if k in get}
        return renamed

    def convert_line(e):
        """
        Convert DXF LINE entities into trimesh Line entities.
        """
        # create a single Line entity
        entities.append(Line(points=len(vertices) + np.arange(2), **info(e)))
        # add the vertices to our collection
        vertices.extend(np.array([[e["10"], e["20"]], [e["11"], e["21"]]], dtype=np.float64))

    def convert_circle(e):
        """
        Convert DXF CIRCLE entities into trimesh Circle entities
        """
        R = float(e["40"])
        C = np.array([e["10"], e["20"]]).astype(np.float64)
        points = to_threepoint(center=C[:2], radius=R)
        entities.append(Arc(points=(len(vertices) + np.arange(3)), closed=True, **info(e)))
        vertices.extend(points)

    def convert_arc(e):
        """
        Convert DXF ARC entities into into trimesh Arc entities.
        """
        # the radius of the circle
        R = float(e["40"])
        # the center point of the circle
        C = np.array([e["10"], e["20"]], dtype=np.float64)
        # the start and end angle of the arc, in degrees
        # this may depend on an AUNITS header data
        A = np.radians(np.array([e["50"], e["51"]], dtype=np.float64))
        # convert center/radius/angle representation
        # to three points on the arc representation
        points = to_threepoint(center=C[:2], radius=R, angles=A)
        # add a single Arc entity
        entities.append(Arc(points=len(vertices) + np.arange(3), closed=False, **info(e)))
        # add the three vertices
        vertices.extend(points)

    def convert_polyline(e):
        """
        Convert DXF LWPOLYLINE entities into trimesh Line entities.
        """
        # load the points in the line
        lines = np.column_stack((e["10"], e["20"])).astype(np.float64)

        # save entity info so we don't have to recompute
        polyinfo = info(e)

        # 70 is the closed flag for polylines
        # if the closed flag is set make sure to close
        is_closed = "70" in e and int(e["70"][0]) & 1
        if is_closed:
            lines = np.vstack((lines, lines[:1]))

        # 42 is the vertex bulge flag for LWPOLYLINE entities
        # "bulge" is autocad for "add a stupid arc using flags
        # in my otherwise normal polygon", it's like SVG arc
        # flags but somehow even more annoying
        if "42" in e:
            # get the actual bulge float values
            bulge = np.array(e["42"], dtype=np.float64)
            # what position were vertices stored at
            vid = np.nonzero(chunk[:, 0] == "10")[0]
            # what position were bulges stored at in the chunk
            bid = np.nonzero(chunk[:, 0] == "42")[0]
            # filter out endpoint bulge if we're not closed
            if not is_closed:
                bid_ok = bid < vid.max()
                bid = bid[bid_ok]
                bulge = bulge[bid_ok]
            # which vertex index is bulge value associated with
            bulge_idx = np.searchsorted(vid, bid)
            # convert stupid bulge to Line/Arc entities
            v, e = bulge_to_arcs(lines=lines, bulge=bulge, bulge_idx=bulge_idx, is_closed=is_closed)
            for i in e:
                # offset added entities by current vertices length
                i.points += len(vertices)
            vertices.extend(v)
            entities.extend(e)
            # done with this polyline
            return

        # we have a normal polyline so just add it
        # as single line entity and vertices
        entities.append(Line(points=np.arange(len(lines)) + len(vertices), **polyinfo))
        vertices.extend(lines)

    def convert_bspline(e):
        """
        Convert DXF Spline entities into trimesh BSpline entities.
        """
        # in the DXF there are n points and n ordered fields
        # with the same group code

        points = np.column_stack((e["10"], e["20"])).astype(np.float64)
        knots = np.array(e["40"]).astype(np.float64)

        # if there are only two points, save it as a line
        if len(points) == 2:
            # create a single Line entity
            entities.append(Line(points=len(vertices) + np.arange(2), **info(e)))
            # add the vertices to our collection
            vertices.extend(points)
            return

        # check bit coded flag for closed
        # closed = bool(int(e['70'][0]) & 1)
        # check euclidean distance to see if closed
        closed = np.linalg.norm(points[0] - points[-1]) < tol.merge

        # create a BSpline entity
        entities.append(BSpline(points=np.arange(len(points)) + len(vertices), knots=knots, closed=closed, **info(e)))
        # add the vertices
        vertices.extend(points)

    def convert_text(e):
        """
        Convert a DXF TEXT entity into a native text entity.
        """
        # text with leading and trailing whitespace removed
        text = e["1"].strip()
        # try getting optional height of text
        try:
            height = float(e["40"])
        except BaseException:
            height = None
        try:
            # rotation angle converted to radians
            angle = np.radians(float(e["50"]))
        except BaseException:
            # otherwise no rotation
            angle = 0.0
        # origin point
        origin = np.array([e["10"], e["20"]], dtype=np.float64)
        # an origin-relative point (so transforms work)
        vector = origin + [np.cos(angle), np.sin(angle)]
        # try to extract a (horizontal, vertical) text alignment
        align = ["center", "center"]
        try:
            align[0] = ["left", "center", "right"][int(e["72"])]
        except BaseException:
            pass
        # append the entity
        entities.append(Text(origin=len(vertices), vector=len(vertices) + 1, height=height, text=text, align=align))
        # append the text origin and direction
        vertices.append(origin)
        vertices.append(vector)

    def convert_insert(e):
        """
        Convert an INSERT entity, which inserts a named group of
        entities (i.e. a "BLOCK") at a specific location.
        """
        if blocks is None:
            return

        # name of block to insert
        name = e["2"]
        # if we haven't loaded the block skip
        if name not in blocks:
            return
        # angle to rotate the block by
        angle = float(e.get("50", 0.0))
        # the insertion point of the block
        offset = np.array([e.get("10", 0.0), e.get("20", 0.0)], dtype=np.float64)
        # what to scale the block by
        scale = np.array([e.get("41", 1.0), e.get("42", 1.0)], dtype=np.float64)

        # the current entities and vertices of the referenced block.
        cv, ce = blocks[name]
        for i in ce:
            # copy the referenced entity as it may be included multiple times
            entities.append(i.copy())
            # offset its vertices to the current index
            entities[-1].points += len(vertices)
        # transform the block's vertices based on the entity settings
        vertices.extend(tf.transform_points(cv, tf.planar_matrix(offset=offset, theta=np.radians(angle), scale=scale)))

    # find the start points of entities
    # DXF object to trimesh object converters
    loaders = {
        "LINE": (dict, convert_line),
        "LWPOLYLINE": (multi_dict, convert_polyline),
        "ARC": (dict, convert_arc),
        "CIRCLE": (dict, convert_circle),
        "SPLINE": (multi_dict, convert_bspline),
        "INSERT": (dict, convert_insert),
        "BLOCK": (dict, convert_insert),
    }

    # store loaded vertices
    vertices = []
    # store loaded entities
    entities = []
    # an old-style polyline entity strings its data across
    # multiple vertex entities like a real asshole
    polyline = None
    # chunks of entities are divided by group-code-0
    inflection = np.nonzero(blob[:, 0] == "0")[0]

    unsupported = defaultdict(lambda: 0)

    # loop through chunks of entity information
    for index in np.array_split(np.arange(len(blob)), inflection):
        # if there is only a header continue
        if len(index) < 1:
            continue
        # chunk will be an (n, 2) array of (group code, data) pairs
        chunk = blob[index]
        # the string representing entity type
        entity_type = chunk[0][1]

        # if we are referencing a block or insert by name make
        # sure the name key is in the original case vs upper-case
        if entity_type in ("BLOCK", "INSERT"):
            try:
                index_name = next(i for i, v in enumerate(chunk) if v[0] == "2")
                chunk[index_name][1] = blob_raw[index][index_name][1]
            except StopIteration:
                pass

        # special case old- style polyline entities
        if entity_type == "POLYLINE":
            polyline = [dict(chunk)]
        # if we are collecting vertex entities
        elif polyline is not None and entity_type == "VERTEX":
            polyline.append(dict(chunk))
        # the end of a polyline
        elif polyline is not None and entity_type == "SEQEND":
            # pull the geometry information for the entity
            lines = np.array([[i["10"], i["20"]] for i in polyline[1:]], dtype=np.float64)

            is_closed = False
            # check for a closed flag on the polyline
            if "70" in polyline[0]:
                # flag is bit- coded integer
                flag = int(polyline[0]["70"])
                # first bit represents closed
                is_closed = bool(flag & 1)
                if is_closed:
                    lines = np.vstack((lines, lines[:1]))

            # get the index of each bulged vertices
            bulge_idx = np.array([i for i, e in enumerate(polyline) if "42" in e], dtype=np.int64)
            # get the actual bulge value
            bulge = np.array([float(e["42"]) for i, e in enumerate(polyline) if "42" in e], dtype=np.float64)
            # convert bulge to new entities
            cv, ce = bulge_to_arcs(lines=lines, bulge=bulge, bulge_idx=bulge_idx, is_closed=is_closed)
            for i in ce:
                # offset entities by existing vertices
                i.points += len(vertices)
            vertices.extend(cv)
            entities.extend(ce)
            # we no longer have an active polyline
            polyline = None
        elif entity_type == "TEXT":
            # text entities need spaces preserved so take
            # group codes from clean representation (0- column)
            # and data from the raw representation (1- column)
            chunk_raw = blob_raw[index]
            # if we didn't use clean group codes we wouldn't
            # be able to access them by key as whitespace
            # is random and crazy, like: '  1 '
            chunk_raw[:, 0] = blob[index][:, 0]
            try:
                convert_text(dict(chunk_raw))
            except BaseException:
                log.debug("failed to load text entity!", exc_info=True)
        # if the entity contains all relevant data we can
        # cleanly load it from inside a single function
        elif entity_type in loaders:
            # the chunker converts an (n,2) list into a dict
            chunker, loader = loaders[entity_type]
            # convert data to dict
            entity_data = chunker(chunk)
            # append data to the lists we're collecting
            loader(entity_data)
        elif entity_type != "ENTITIES":
            unsupported[entity_type] += 1
    if len(unsupported) > 0:
        log.debug("skipping dxf entities: {}".format(", ".join("{}: {}".format(k, v) for k, v in unsupported.items())))
    # stack vertices into single array
    vertices = util.vstack_empty(vertices).astype(np.float64)
    if return_name:
        name = blob_raw[blob[:, 0] == "2"][0][1]
        return vertices, entities, name

    return vertices, entities


def export_dxf(path, only_layers=None):
    """
    Export a 2D path object to a DXF file.

    Parameters
    ----------
    path : trimesh.path.path.Path2D
      Input geometry to export
    only_layers : None or set
      If passed only export the layers specified

    Returns
    ----------
    export : str
      Path formatted as a DXF file
    """
    # get the template for exporting DXF files
    template = resources.get("templates/dxf.json", decode_json=True)

    def format_points(points, as_2D=False, increment=True):
        """
        Format points into DXF- style point string.

        Parameters
        -----------
        points : (n,2) or (n,3) float
          Points in space
        as_2D : bool
          If True only output 2 points per vertex
        increment : bool
          If True increment group code per point
          Example:
            [[X0, Y0, Z0], [X1, Y1, Z1]]
          Result, new lines replaced with spaces:
            True  -> 10 X0 20 Y0 30 Z0 11 X1 21 Y1 31 Z1
            False -> 10 X0 20 Y0 30 Z0 10 X1 20 Y1 30 Z1

        Returns
        -----------
        packed : str
          Points formatted with group code
        """
        points = np.asanyarray(points, dtype=np.float64)
        # get points in 3D
        three = util.stack_3D(points)
        if increment:
            group = np.tile(np.arange(len(three), dtype=np.int64).reshape((-1, 1)), (1, 3))
        else:
            group = np.zeros((len(three), 3), dtype=np.int64)
        group += [10, 20, 30]

        if as_2D:
            group = group[:, :2]
            three = three[:, :2]
        # join into result string
        packed = "\n".join("{:d}\n{:.12g}".format(g, v) for g, v in zip(group.reshape(-1), three.reshape(-1)))

        return packed

    def entity_info(entity):
        """
        Pull layer, color, and name information about an entity

        Parameters
        -----------
        entity : entity object
          Source entity to pull metadata

        Returns
        ----------
        subs : dict
          Has keys 'COLOR', 'LAYER', 'NAME'
        """
        # TODO : convert RGBA entity.color to index
        subs = {"COLOR": 255, "LAYER": 0, "NAME": str(id(entity))[:16]}  # default is ByLayer
        if hasattr(entity, "layer"):
            # make sure layer name is forced into ASCII
            subs["LAYER"] = util.to_ascii(entity.layer)
        return subs

    def convert_line(line, vertices):
        """
        Convert an entity to a discrete polyline

        Parameters
        -------------
        line : entity
          Entity which has 'e.discrete' method
        vertices : (n, 2) float
          Vertices in space

        Returns
        -----------
        as_dxf : str
          Entity exported as a DXF
        """
        # get a discrete representation of entity
        points = line.discrete(vertices)
        # if one or fewer points return nothing
        if len(points) <= 1:
            return ""

        # generate a substitution dictionary for template
        subs = entity_info(line)
        subs["POINTS"] = format_points(points, as_2D=True, increment=False)
        subs["TYPE"] = "LWPOLYLINE"
        subs["VCOUNT"] = len(points)
        # 1 is closed
        # 0 is default (open)
        subs["FLAG"] = int(bool(line.closed))

        result = template["line"].format(**subs)
        return result

    def convert_arc(arc, vertices):
        # get the center of arc and include span angles
        info = arc.center(vertices, return_angle=True, return_normal=False)
        subs = entity_info(arc)
        center = info["center"]
        if len(center) == 2:
            center = np.append(center, 0.0)
        data = "10\n{:.12g}\n20\n{:.12g}\n30\n{:.12g}".format(*center)
        data += "\n40\n{:.12g}".format(info["radius"])

        if arc.closed:
            subs["TYPE"] = "CIRCLE"
        else:
            subs["TYPE"] = "ARC"
            # an arc is the same as a circle, with an added start
            # and end angle field
            data += "\n100\nAcDbArc"
            data += "\n50\n{:.12g}\n51\n{:.12g}".format(*np.degrees(info["angles"]))
        subs["DATA"] = data
        result = template["arc"].format(**subs)

        return result

    def convert_bspline(spline, vertices):
        # points formatted with group code
        points = format_points(vertices[spline.points], increment=False)

        # (n,) float knots, formatted with group code
        knots = ("40\n{:.12g}\n" * len(spline.knots)).format(*spline.knots)[:-1]

        # bit coded
        flags = {"closed": 1, "periodic": 2, "rational": 4, "planar": 8, "linear": 16}

        flag = flags["planar"]
        if spline.closed:
            flag = flag | flags["closed"]

        normal = [0.0, 0.0, 1.0]
        n_code = [210, 220, 230]
        n_str = "\n".join("{:d}\n{:.12g}".format(i, j) for i, j in zip(n_code, normal))

        subs = entity_info(spline)
        subs.update(
            {
                "TYPE": "SPLINE",
                "POINTS": points,
                "KNOTS": knots,
                "NORMAL": n_str,
                "DEGREE": 3,
                "FLAG": flag,
                "FCOUNT": 0,
                "KCOUNT": len(spline.knots),
                "PCOUNT": len(spline.points),
            }
        )
        # format into string template
        result = template["bspline"].format(**subs)

        return result

    def convert_text(txt, vertices):
        """
        Convert a Text entity to DXF string.
        """
        # start with layer info
        sub = entity_info(txt)
        # get the origin point of the text
        sub["ORIGIN"] = format_points(vertices[[txt.origin]], increment=False)
        # rotation angle in degrees
        sub["ANGLE"] = np.degrees(txt.angle(vertices))
        # actual string of text with spaces escaped
        # force into ASCII to avoid weird encoding issues
        sub["TEXT"] = txt.text.replace(" ", _SAFESPACE).encode("ascii", errors="ignore").decode("ascii")
        # height of text
        sub["HEIGHT"] = txt.height
        result = template["text"].format(**sub)
        return result

    def convert_brief(entity, vertices):
        """
        For entities we don't know how to handle, return their
        discrete form as a polyline
        """
        return convert_line(entity, vertices)

    # make sure we're not losing a ton of
    # precision in the string conversion
    np.set_printoptions(precision=12)
    # trimesh entity to DXF entity converters
    conversions = {
        "Line": convert_line,
        "Text": convert_text,
        "Arc": convert_arc,
        "Bezier": convert_brief,
        "BSpline": convert_bspline,
    }
    collected = []
    for e, layer in zip(path.entities, path.layers):
        name = type(e).__name__
        # only export specified layers
        if only_layers is not None and layer not in only_layers:
            continue
        if name in conversions:
            converted = conversions[name](e, path.vertices).strip()
            if len(converted) > 0:
                # only save if we converted something
                collected.append(converted)
        else:
            log.debug("Entity type %s not exported!", name)

    # join all entities into one string
    entities_str = "\n".join(collected)

    # add in the extents of the document as explicit XYZ lines
    hsub = {"EXTMIN_{}".format(k): v for k, v in zip("XYZ", np.append(path.bounds[0], 0.0))}
    hsub.update({"EXTMAX_{}".format(k): v for k, v in zip("XYZ", np.append(path.bounds[1], 0.0))})
    # apply a units flag defaulting to `1`
    hsub["LUNITS"] = _UNITS_TO_DXF.get(path.units, 1)
    # run the format for the header
    sections = [template["header"].format(**hsub).strip()]
    # do the same for entities
    sections.append(template["entities"].format(ENTITIES=entities_str).strip())
    # and the footer
    sections.append(template["footer"].strip())

    # filter out empty sections
    # random whitespace causes AutoCAD to fail to load
    # although Draftsight, LibreCAD, and Inkscape don't care
    # what a giant legacy piece of shit
    # create the joined string blob
    blob = "\n".join(sections).replace(_SAFESPACE, " ")
    # run additional self- checks
    if tol.strict:
        # check that every line pair is (group code, value)
        lines = str.splitlines(str(blob))
        # should be even number of lines
        assert (len(lines) % 2) == 0
        # group codes should all be convertible to int and positive
        assert all(int(i) >= 0 for i in lines[::2])
        # make sure we didn't slip any unicode in there
        blob.encode("ascii")

    return blob


def bulge_to_arcs(lines, bulge, bulge_idx, is_closed=False, metadata=None):
    """
    Polylines can have "vertex bulge," which means the polyline
    has an arc tangent to segments, rather than meeting at a
    vertex.

    From Autodesk reference:
    The bulge is the tangent of one fourth the included
    angle for an arc segment, made negative if the arc
    goes clockwise from the start point to the endpoint.
    A bulge of 0 indicates a straight segment, and a
    bulge of 1 is a semicircle.

    Parameters
    ----------------
    lines : (n, 2) float
      Polyline vertices in order
    bulge : (m,) float
      Vertex bulge value
    bulge_idx : (m,) float
      Which index of lines is bulge associated with
    is_closed : bool
      Is segment closed
    metadata : None, or dict
      Entity metadata to add

    Returns
    ---------------
    vertices : (a, 2) float
      New vertices for poly-arc
    entities : (b,) entities.Entity
      New entities, either line or arc
    """
    # make sure lines are 2D array
    lines = np.asanyarray(lines, dtype=np.float64)

    # make sure inputs are numpy arrays
    bulge = np.asanyarray(bulge, dtype=np.float64)
    bulge_idx = np.asanyarray(bulge_idx, dtype=np.int64)

    # filter out zero- bulged polylines
    ok = np.abs(bulge) > 1e-5
    bulge = bulge[ok]
    bulge_idx = bulge_idx[ok]

    # metadata to apply to new entities
    if metadata is None:
        metadata = {}

    # if there's no bulge, just return the input curve
    if len(bulge) == 0:
        index = np.arange(len(lines))
        # add a single line entity and vertices
        entities = [Line(index, **metadata)]
        return lines, entities

    # use bulge to calculate included angle of the arc
    angle = np.arctan(bulge) * 4.0
    # the indexes making up a bulged segment
    tid = np.column_stack((bulge_idx, bulge_idx - 1))
    # if it's a closed segment modulus to start vertex
    if is_closed:
        tid %= len(lines)

    # the vector connecting the two ends of the arc
    vector = lines[tid[:, 0]] - lines[tid[:, 1]]

    # the length of the connector segment
    length = np.linalg.norm(vector, axis=1)

    # perpendicular vectors by crossing vector with Z
    perp = np.cross(np.column_stack((vector, np.zeros(len(vector)))), np.ones((len(vector), 3)) * [0, 0, 1])
    # strip the zero Z
    perp = util.unitize(perp[:, :2])

    # midpoint of each line
    midpoint = lines[tid].mean(axis=1)

    # calculate the signed radius of each arc segment
    radius = (length / 2.0) / np.sin(angle / 2.0)

    # offset magnitude to point on arc
    offset = radius - np.cos(angle / 2) * radius

    # convert each arc to three points:
    # start, any point on arc, end
    three = np.column_stack((lines[tid[:, 0]], midpoint + perp * offset.reshape((-1, 1)), lines[tid[:, 1]])).reshape(
        (-1, 3, 2)
    )

    # if we're in strict mode make sure our arcs
    # have the same magnitude as the input data
    if tol.strict:
        from ..arc import arc_center

        check_angle = [arc_center(i)["span"] for i in three]
        assert np.allclose(np.abs(angle), np.abs(check_angle))

        check_radii = [arc_center(i)["radius"] for i in three]
        assert np.allclose(check_radii, np.abs(radius))

    # collect new entities and vertices
    entities, vertices = [], []
    # add the entities for each new arc
    for arc_points in three:
        entities.append(Arc(points=np.arange(3) + len(vertices), **metadata))
        vertices.extend(arc_points)

    # if there are unconsumed line
    # segments add them to drawing
    if (len(lines) - 1) > len(bulge):
        # indexes of line segments
        existing = util.stack_lines(np.arange(len(lines)))
        # remove line segments replaced with arcs
        for line_idx in grouping.boolean_rows(existing, np.sort(tid, axis=1), np.setdiff1d):
            # add a single line entity and vertices
            entities.append(Line(points=np.arange(2) + len(vertices), **metadata))
            vertices.extend(lines[line_idx].copy())

    # make sure vertices are clean numpy array
    vertices = np.array(vertices, dtype=np.float64)

    return vertices, entities


def get_key(blob, field, code):
    """
    Given a loaded (n, 2) blob and a field name
    get a value by code.
    """
    try:
        line = blob[np.nonzero(blob[:, 1] == field)[0][0] + 1]
    except IndexError:
        return None
    if line[0] == code:
        try:
            return int(line[1])
        except ValueError:
            return line[1]
    else:
        return None


# store the loaders we have available
_dxf_loaders = {"dxf": load_dxf}
