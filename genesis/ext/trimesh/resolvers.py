"""
resolvers.py
---------------

Provides a common interface to load assets referenced by name
like MTL files, texture images, etc. Assets can be from ZIP
archives, web assets, or a local file path.
"""

import os
import abc
import itertools

from . import util
from . import caching

# URL parsing for remote resources via WebResolver
try:
    # Python 3
    from urllib.parse import urlparse
except ImportError:
    # Python 2
    from urlparse import urlparse


class Resolver(util.ABC):
    """
    The base class for resolvers.
    """

    @abc.abstractmethod
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("Use a resolver subclass!")

    @abc.abstractmethod
    def get(self, key):
        raise NotImplementedError()

    @abc.abstractmethod
    def write(self, name, data):
        raise NotImplementedError("`write` not implemented!")

    @abc.abstractmethod
    def namespaced(self, namespace):
        raise NotImplementedError("`namespaced` not implemented!")

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.write(key, value)

    def __contains__(self, key):
        return key in self.keys()


class FilePathResolver(Resolver):
    """
    Resolve files from a source path on the file system.
    """

    def __init__(self, source):
        """
        Resolve files based on a source path.

        Parameters
        ------------
        source : str
          File path where mesh was loaded from
        """
        # remove everything other than absolute path
        clean = os.path.expanduser(os.path.abspath(str(source)))

        self.clean = clean
        if os.path.isdir(clean):
            # if we were passed a directory use it
            self.parent = clean
        else:
            # otherwise get the parent directory we've been passed
            split = os.path.split(clean)
            self.parent = split[0]

        # exit if directory doesn't exist
        if not os.path.isdir(self.parent):
            raise ValueError("path `{} `not a directory!".format(self.parent))

    def keys(self):
        """
        List all files available to be loaded.

        Yields
        -----------
        name : str
          Name of a file which can be accessed.
        """
        parent = self.parent
        for path, _, names in os.walk(self.parent):
            # strip any leading parent key
            if path.startswith(parent):
                path = path[len(parent) :]
            # yield each name
            for name in names:
                yield os.path.join(path, name)

    def namespaced(self, namespace):
        """
        Return a resolver which changes the root of the
        resolver by an added namespace.

        Parameters
        -------------
        namespace : str
          Probably a subdirectory

        Returns
        --------------
        resolver : FilePathResolver
          Resolver with root directory changed.
        """
        return FilePathResolver(os.path.join(self.parent, namespace))

    def get(self, name):
        """
        Get an asset.

        Parameters
        -------------
        name : str
          Name of the asset

        Returns
        ------------
        data : bytes
          Loaded data from asset
        """
        # load the file by path name
        path = os.path.join(self.parent, name.strip())
        if not os.path.exists(path):
            path = os.path.join(self.parent, os.path.split(name)[-1])
        with open(path, "rb") as f:
            data = f.read()
        return data

    def write(self, name, data):
        """
        Write an asset to a file path.

        Parameters
        -----------
        name : str
          Name of the file to write
        data : str or bytes
          Data to write to the file
        """
        # write files to path name
        with open(os.path.join(self.parent, name.strip()), "wb") as f:
            # handle encodings correctly for str/bytes
            util.write_encoded(file_obj=f, stuff=data)


class ZipResolver(Resolver):
    """
    Resolve files inside a ZIP archive.
    """

    def __init__(self, archive=None, namespace=None):
        """
        Resolve files inside a ZIP archive as loaded by
        trimesh.util.decompress

        Parameters
        -------------
        archive : dict
          Contains resources as file object
        namespace : None or str
          If passed will only show keys that start
          with this value and this substring must be
          removed for any get calls.
        """
        self.archive = archive
        if isinstance(namespace, str):
            self.namespace = namespace.strip().rstrip("/") + "/"
        else:
            self.namespace = None

    def keys(self):
        """
        Get the available keys in the current archive.

        Returns
        -----------
        keys : iterable
          Keys in the current archive.
        """
        if self.namespace is not None:
            namespace = self.namespace
            length = len(namespace)
            # only return keys that start with the namespace
            # and strip off the namespace from the returned
            # keys.
            return [k[length:] for k in self.archive.keys() if k.startswith(namespace) and len(k) > length]
        return self.archive.keys()

    def write(self, key, value):
        """
        Store a value in the current archive.

        Parameters
        -----------
        key : hashable
          Key to store data under.
        value : str, bytes, file-like
          Value to store.
        """
        if self.archive is None:
            self.archive = {}
        self.archive[key] = value

    def get(self, name):
        """
        Get an asset from the ZIP archive.

        Parameters
        -------------
        name : str
          Name of the asset

        Returns
        -------------
        data : bytes
          Loaded data from asset
        """
        # not much we can do with None
        if name is None:
            return
        # make sure name is a string
        if hasattr(name, "decode"):
            name = name.decode("utf-8")
        # store reference to archive inside this function
        archive = self.archive
        # requested name not identical in
        # storage so attempt to recover
        if name not in archive:
            # loop through unique results
            for option in nearby_names(name, self.namespace):
                if option in archive:
                    # cleaned option is in archive
                    # so store value and exit
                    name = option
                    break

        # get the stored data
        obj = archive[name]
        # if the dict is storing data as bytes just return
        if isinstance(obj, (bytes, str)):
            return obj
        # otherwise get it as a file object
        # read file object from beginning
        obj.seek(0)
        # data is stored as a file object
        data = obj.read()
        obj.seek(0)
        return data

    def namespaced(self, namespace):
        """
        Return a "sub-resolver" with a root namespace.

        Parameters
        -------------
        namespace : str
          The root of the key to clip off, i.e. if
          this resolver has key `a/b/c` you can get
          'a/b/c' with resolver.namespaced('a/b').get('c')

        Returns
        -----------
        resolver : Resolver
          Namespaced resolver.
        """
        return ZipResolver(archive=self.archive, namespace=namespace)

    def export(self):
        """
        Export the contents of the current archive as
        a ZIP file.

        Returns
        ------------
        compressed : bytes
          Compressed data in ZIP format.
        """
        return util.compress(self.archive)


class WebResolver(Resolver):
    """
    Resolve assets from a remote URL.
    """

    def __init__(self, url):
        """
        Resolve assets from a base URL.

        Parameters
        --------------
        url : str
          Location where a mesh was stored or
          directory where mesh was stored
        """
        if hasattr(url, "decode"):
            url = url.decode("utf-8")

        # parse string into namedtuple
        parsed = urlparse(url)

        # we want a base url
        split = [i for i in parsed.path.split("/") if len(i) > 0]

        # if the last item in the url path is a filename
        # move up a "directory" for the base path
        if len(split) == 0:
            path = ""
        elif "." in split[-1]:
            # clip off last item
            path = "/".join(split[:-1])
        else:
            # recombine into string ignoring any double slashes
            path = "/".join(split)
        self.base_url = (
            "/".join(i for i in [parsed.scheme + ":/", parsed.netloc.strip("/"), path.strip("/")] if len(i) > 0) + "/"
        )

        # our string handling should have never inserted double slashes
        assert "//" not in self.base_url[len(parsed.scheme) + 3 :]
        # we should always have ended with a single slash
        assert self.base_url.endswith("/")

    def get(self, name):
        """
        Get a resource from the remote site.

        Parameters
        -------------
        name : str
          Asset name, i.e. 'quadknot.obj.mtl'
        """
        # do import here to keep soft dependency
        import requests

        # remove leading and trailing whitespace
        name = name.strip()
        # fetch the data from the remote url

        # base url has been carefully formatted
        url = self.base_url + name

        response = requests.get(url)

        if response.status_code != 200:
            # try to strip off filesystem crap
            if name.startswith("./"):
                name = name[2:]
            response = requests.get(self.base_url + name)

        if response.status_code == "404":
            raise ValueError(response.content)

        # return the bytes of the response
        return response.content

    def namespaced(self, namespace):
        """
        Return a namespaced version of current resolver.

        Parameters
        -------------
        namespace : str
          URL fragment

        Returns
        -----------
        resolver : WebResolver
          With sub-url: `https://example.com/{namespace}`
        """
        # join the base url and the namespace
        return WebResolver(url=self.base_url + namespace)

    def write(self, key, value):
        raise NotImplementedError("can't write to remote")


class GithubResolver(Resolver):
    def __init__(self, repo, branch=None, commit=None, save=None):
        """
        Get files from a remote Github repository by
        downloading a zip file with the entire branch
        or a specific commit.

        Parameters
        -------------
        repo : str
          In the format of `owner/repo`
        branch : str
          The remote branch you want to get files from.
        commit : str
          The full commit hash: pass either this OR branch.
        save : None or str
          A path if you want to save results locally.
        """
        # the github URL for the latest commit of a branch.
        if commit is None:
            self.url = ("https://github.com/{repo}/archive/" + "refs/heads/{branch}.zip").format(
                repo=repo, branch=branch
            )
        else:
            # get a commit URL
            self.url = ("https://github.com/{repo}/archive/" + "{commit}.zip").format(repo=repo, commit=commit)

        if save is not None:
            self.cache = caching.DiskCache(save)
        else:
            self.cache = None

    def keys(self):
        """
        List the available files in the repository.

        Returns
        ----------
        keys : iterable
          Keys available to the resolved.
        """
        return self.zipped.keys()

    def write(self, name, data):
        raise NotImplementedError("`write` not implemented!")

    @property
    def zipped(self):
        """

        - opened zip file
        - locally saved zip file
        - retrieve zip file and saved
        """

        def fetch():
            """
            Fetch the remote zip file.
            """
            import requests

            response = requests.get(self.url)
            if not response.ok:
                raise ValueError(response.content)
            return response.content

        if hasattr(self, "_zip"):
            return self._zip
        # download the archive or get from disc
        raw = self.cache.get(self.url, fetch)
        # create a zip resolver for the archive
        self._zip = ZipResolver(util.decompress(util.wrap_as_stream(raw), file_type="zip"))

        return self._zip

    def get(self, key):
        return self.zipped.get(key)

    def namespaced(self, namespace):
        """
        Return a "sub-resolver" with a root namespace.

        Parameters
        -------------
        namespace : str
          The root of the key to clip off, i.e. if
          this resolver has key `a/b/c` you can get
          'a/b/c' with resolver.namespaced('a/b').get('c')

        Returns
        -----------
        resolver : Resolver
          Namespaced resolver.
        """
        return self.zipped.namespaced(namespace)


def nearby_names(name, namespace=None):
    """
    Try to find nearby variants of a specified name.

    Parameters
    ------------
    name : str
      Initial name.

    Yields
    -----------
    nearby : str
      Name that is a lightly permutated version
      of the initial name.
    """

    # the various operations that *might* result in a correct key
    def trim(prefix, item):
        if item.startswith(prefix):
            return item[len(prefix) :]
        return item

    cleaners = [
        lambda x: x,
        lambda x: x.strip(),
        lambda x: trim("./", x),
        lambda x: trim(".\\", x),
        lambda x: trim("\\", x),
        lambda x: os.path.split(x)[-1],
        lambda x: x.replace("%20", " "),
    ]

    if namespace is None:
        namespace = ""

    # make sure we don't return repeat values
    hit = set()
    for f in cleaners:
        # try just one cleaning function
        current = f(name)
        if current in hit:
            continue
        hit.add(current)
        yield namespace + current

    for a, b in itertools.combinations(cleaners, 2):
        # apply both clean functions
        current = a(b(name))
        if current in hit:
            continue
        hit.add(current)
        yield namespace + current

        # try applying in reverse order
        current = b(a(name))
        if current in hit:
            continue
        hit.add(current)
        yield namespace + current

    if ".." in name and namespace is not None:
        # if someone specified relative paths give it one attempt
        strip = namespace.strip("/").split("/")[: -name.count("..")]
        strip.extend(name.split("..")[-1].strip("/").split("/"))
        yield "/".join(strip)
