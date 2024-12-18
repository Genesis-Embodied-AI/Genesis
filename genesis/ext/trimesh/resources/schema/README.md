# trimesh/resources/schemas

Contain schemas for formats when available. They are currently mostly [JSON schema](https://json-schema.org/) although if formats have an XSD, DTD, or other schema format we are happy to include it here.

The `primitive` schema directory is a [JSON schema](https://json-schema.org/) for `trimesh` exports. The goal is if we implement a `to_dict` method to have a well-defined schema we can validate in unit tests.
