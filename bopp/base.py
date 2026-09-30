from typing import Annotated, get_args, get_origin

import msgspec
import pyarrow as pa

_DTYPES = {
    "float32": pa.float32(),
    "float64": pa.float64(),
    "int8": pa.int8(),
    "int16": pa.int16(),
    "int32": pa.int32(),
    "int64": pa.int64(),
    "uint8": pa.uint8(),
    "uint16": pa.uint16(),
    "uint32": pa.uint32(),
    "uint64": pa.uint64(),
    "bool": pa.bool_(),
    "string": pa.string(),
}


def _declared_dtype(type_hint) -> pa.DataType | None:
    """The Arrow dtype named in an ``Annotated[Array, "<dtype>"]`` hint, if any."""
    while get_origin(type_hint) is Annotated:
        for metadata in type_hint.__metadata__:
            if isinstance(metadata, str) and metadata in _DTYPES:
                return _DTYPES[metadata]
        type_hint = get_args(type_hint)[0]
    return None


def apply_declared_dtypes(struct: msgspec.Struct) -> None:
    """
    Cast every Array field to the dtype its schema declares, recursing into nested
    structs (coordinates, fraction buffers). msgspec hands dec_hook the bare
    ``pyarrow.Array`` type without its Annotated metadata, so the precision hints
    can only be honoured after decoding. A safe cast refuses lossy values, e.g. a
    non-integral position in a discrete unit such as samples.
    """
    for field in msgspec.structs.fields(type(struct)):
        value = getattr(struct, field.name)
        if isinstance(value, msgspec.Struct):
            apply_declared_dtypes(value)
            continue
        dtype = _declared_dtype(field.type)
        if dtype is None or not isinstance(value, pa.Array) or value.type == dtype:
            continue
        try:
            setattr(struct, field.name, value.cast(dtype, safe=True))
        except pa.ArrowInvalid as error:
            raise ValueError(f"{type(struct).__name__}.{field.name} must be {dtype}: {error}") from None


class BoppBase(msgspec.Struct):
    """
    A base class for BOPP models. Dynamically validates parallel columnar
    arrays on the root Annotation node, safely ignoring sub-models.
    """
    def __post_init__(self):
        # GUARD: Only run this on top-level Annotation containers
        if not hasattr(self, "payload"):
            return

        apply_declared_dtypes(self)

        all_lengths = []

        extent_field = getattr(self, "extent", msgspec.UNSET)
        if extent_field is not msgspec.UNSET and extent_field is not None:
            extent_lengths = self._get_all_column_lengths(extent_field, "Extent")
            all_lengths.extend(extent_lengths)
            self._check_shared_unit(extent_field)

        payload_field = getattr(self, "payload", msgspec.UNSET)
        if payload_field is not msgspec.UNSET and payload_field is not None:
            payload_lengths = self._get_all_column_lengths(payload_field, "Payload")
            all_lengths.extend(payload_lengths)

        confidence_field = getattr(self, "confidence", msgspec.UNSET)
        if confidence_field is not msgspec.UNSET and confidence_field is not None:
            conf_lengths = self._get_all_column_lengths(confidence_field, "Confidence")
            all_lengths.extend(conf_lengths)

        # Ensure all found arrays have the same length
        if len(set(all_lengths)) > 1:
            raise ValueError(
                f"Length mismatch: Found multiple array lengths {set(all_lengths)} "
                "across Extent, Payload, and Confidence facets."
            )

    @staticmethod
    def _get_all_column_lengths(facet_struct: msgspec.Struct, facet_name: str) -> list[int]:
        """
        Dynamically scans a msgspec struct for all list/array fields, descending
        into nested structs (coordinates, fraction buffers), and returns their lengths.
        """
        lengths = []

        def visit(struct: msgspec.Struct) -> None:
            for field in msgspec.structs.fields(type(struct)):
                val = getattr(struct, field.name)
                if isinstance(val, (pa.Array, list)):
                    lengths.append(len(val))
                elif isinstance(val, msgspec.Struct):
                    visit(val)

        visit(facet_struct)

        if not lengths:
            raise ValueError(f"{facet_name} struct ({type(facet_struct).__name__}) contains no lists to measure.")

        return lengths

    @staticmethod
    def _check_shared_unit(extent_struct: msgspec.Struct) -> None:
        """All coordinates of one extent must be measured in the same unit."""
        units = {
            getattr(val, "__struct_config__").tag
            for field in msgspec.structs.fields(type(extent_struct))
            if isinstance(val := getattr(extent_struct, field.name), msgspec.Struct)
            and getattr(val, "__struct_config__").tag_field == "unit"
        }
        if len(units) > 1:
            raise ValueError(f"Unit mismatch: extent mixes coordinates in {sorted(units)}.")
