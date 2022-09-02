from .module import Module
from typing import Any, Optional, Union, overload, TypeVar, Iterable, Tuple, Mapping, Iterator, KeysView, ValuesView, \
    ItemsView, Sequence
from collections import OrderedDict
from ... import Tensor
from .. import Parameter


class Container(Module):
    def __init__(self, **kwargs: Any) -> None: ...


T = TypeVar('T')


class Sequential(Module):
    @overload
    def __init__(self, *args: Module) -> None: ...

    @overload
    def __init__(self, arg: OrderedDict[str, Module]) -> None: ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self: T, idx: slice) -> T: ...

    def __setitem__(self, idx: Union[int], module: Module) -> None: ...

    def __delitem__(self, idx: Union[slice, int]) -> None: ...

    def __len__(self) -> int: ...

    def forward(self, input: Tensor) -> Tensor: ...  # type: ignore

    def __call__(self, input: Tensor) -> Tensor: ...  # type: ignore


class ModuleList(Module, Sequence[Module]):
    def __init__(self, modules: Optional[Iterable[Module]] = ...) -> None: ...

    @overload
    def __getitem__(self, idx: int) -> Module: ...

    @overload
    def __getitem__(self: T, idx: slice) -> T: ...

    def __setitem__(self, idx: int, module: Module) -> None: ...

    def __delitem__(self, idx: Union[int, slice]) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Module]: ...

    def __iadd__(self: T, modules: Iterable[Module]) -> T: ...

    def insert(self, index: int, module: Module) -> None: ...

    def append(self: T, module: Module) -> T: ...

    def extend(self: T, modules: Iterable[Module]) -> T: ...


class ModuleDict(Module):
    def __init__(self, modules: Optional[Mapping[str, Module]] = ...) -> None: ...

    def __getitem__(self, key: str): ...

    def __setitem__(self, key: str, module: Module) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[str]: ...

    def __contains__(self, key: str) -> bool: ...

    def clear(self) -> None: ...

    def pop(self, key: str): ...

    def keys(self) -> KeysView[str]: ...

    def items(self) -> ItemsView[str, Module]: ...

    def values(self) -> ValuesView[Module]: ...

    def update(self, modules: Mapping[str, Module]) -> None: ...


class ParameterList(Module, Sequence[Parameter]):
    def __init__(self, parameters: Optional[Iterable[Parameter]] = ...) -> None: ...

    @overload
    def __getitem__(self, idx: int) -> Parameter: ...

    @overload
    def __getitem__(self: T, idx: slice) -> T: ...

    def __setitem__(self, idx: int, param: Parameter) -> None: ...

    def __delitem__(self, idx: Union[int, slice]) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[Parameter]: ...

    def __iadd__(self: T, parameters: Iterable[Parameter]) -> T: ...

    def insert(self, index: int, parameter: Parameter) -> None: ...

    def append(self: T, parameter: Parameter) -> T: ...

    def extend(self: T, parameters: Iterable[Parameter]) -> T: ...


class ParameterDict(Module):
    def __init__(self, parameters: Optional[Mapping[str, Parameter]] = ...) -> None: ...

    def __getitem__(self, key: str): ...

    def __setitem__(self, key: str, param: Parameter) -> None: ...

    def __delitem__(self, key: str) -> None: ...

    def __len__(self) -> int: ...

    def __iter__(self) -> Iterator[str]: ...

    def __contains__(self, key: str) -> bool: ...

    def clear(self) -> None: ...

    def pop(self, key: str): ...

    def keys(self) -> KeysView[str]: ...

    def items(self) -> ItemsView[str, Parameter]: ...

    def values(self) -> ValuesView[Parameter]: ...

    def update(self, parameters: Mapping[str, Parameter]) -> None: ...
