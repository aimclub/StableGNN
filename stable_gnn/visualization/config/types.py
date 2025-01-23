import typing as t

TEdgeList = t.NewType("TEdgeList", list[tuple[int, int]])
TGraphEdgeList = t.NewType("TGraphEdgeList", tuple[TEdgeList, list[float]])
TVectorCoordinates = t.NewType("TVectorCoordinates", list[tuple[float, float]])
