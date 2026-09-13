"""Tests for the unified transform row model: the four ``TransformFunction``/``TransformCodec``
pairs (``infrastructure/transform/records.py``), the one ``TransformTable`` lookup-and-apply
class, composition, and the local-scale helper (``application/coordinate_transforms.py``).

numpy-only — no cv2 / model downloads. The homography matrices are built directly here
(the cv2 fit lives in ``test_world_calibration.py``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tratrac.application.coordinate_transforms import (
	TransformTable,
	TranslationTransform,
	compose,
	compose_invertible,
	local_scale_at,
)
from tratrac.domain.geometry import Point2D, Transform2D
from tratrac.infrastructure.transform.records import (
	HomographyCodec,
	HomographyFunction,
	IdentityCodec,
	IdentityFunction,
	ScaleCodec,
	ScaleFunction,
	SimilarityCodec,
	SimilarityFunction,
	TransformRow,
	decode_row,
	encode_row,
)


def _square(x0: float, y0: float, x1: float, y1: float) -> tuple[Point2D, ...]:
	return (Point2D(x0, y0), Point2D(x1, y0), Point2D(x1, y1), Point2D(x0, y1))


# Strictly contains every point queried against it below (point_in_polygon excludes
# the boundary, so this can't start flush at 0,0).
_WIDE_ZONE = _square(-1000.0, -1000.0, 1000.0, 1000.0)


def _scale_matrix(s: float) -> tuple[float, ...]:
	"""A pure-scaling (s metres per pixel) homography: world = s * image."""
	return (s, 0.0, 0.0, 0.0, s, 0.0, 0.0, 0.0, 1.0)


class TestIdentityFunction:
	def test_apply_and_reverse_are_no_ops(self) -> None:
		function = IdentityFunction()
		assert function.apply(Point2D(3.0, 4.0)) == Point2D(3.0, 4.0)
		assert function.reverse(Point2D(3.0, 4.0)) == Point2D(3.0, 4.0)


class TestScaleFunction:
	def test_apply_multiplies_both_axes(self) -> None:
		function = ScaleFunction(0.5)
		assert function.apply(Point2D(10.0, 20.0)) == Point2D(5.0, 10.0)

	def test_reverse_divides_back_out(self) -> None:
		function = ScaleFunction(0.5)
		original = Point2D(10.0, 20.0)
		assert function.reverse(function.apply(original)) == original

	def test_rejects_a_non_positive_scale(self) -> None:
		with pytest.raises(ValueError, match="positive"):
			ScaleFunction(0.0)


class TestSimilarityFunction:
	def test_apply_delegates_to_the_transform(self) -> None:
		t = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=2.0, ty=3.0)
		function = SimilarityFunction(t)
		assert function.apply(Point2D(1.0, 1.0)) == t.apply(Point2D(1.0, 1.0))

	def test_reverse_round_trips(self) -> None:
		t = Transform2D(a=2.0, b=0.3, tx=5.0, c=-0.1, d=1.5, ty=-2.0)
		function = SimilarityFunction(t)
		original = Point2D(37.0, -12.0)
		recovered = function.reverse(function.apply(original))
		assert recovered.x == pytest.approx(original.x, abs=1e-9)
		assert recovered.y == pytest.approx(original.y, abs=1e-9)


class TestHomographyFunction:
	def test_identity_matrix_is_a_no_op(self) -> None:
		function = HomographyFunction(tuple(np.eye(3, dtype=np.float64).flatten()))
		assert function.apply(Point2D(7.0, 9.0)) == Point2D(7.0, 9.0)

	def test_pure_scale_multiplies_both_axes(self) -> None:
		function = HomographyFunction(_scale_matrix(0.5))
		assert function.apply(Point2D(10.0, 20.0)) == Point2D(5.0, 10.0)

	def test_perspective_divide_is_applied(self) -> None:
		# Last row (0, 0.1, 1): w = 0.1*y + 1 -> a real projective divide, not affine.
		matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.1, 1.0)
		out = HomographyFunction(matrix).apply(Point2D(2.0, 10.0))
		# w = 0.1*10 + 1 = 2 -> (2/2, 10/2)
		assert out.x == pytest.approx(1.0)
		assert out.y == pytest.approx(5.0)

	def test_point_at_infinity_raises(self) -> None:
		# Last row (0, 1, -5): w = y - 5 = 0 at y = 5.
		matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -5.0)
		with pytest.raises(ValueError, match="infinity"):
			HomographyFunction(matrix).apply(Point2D(3.0, 5.0))

	def test_reverse_round_trips_through_a_general_homography(self) -> None:
		# Not a pure scale/affine -- includes a real perspective term, so a correct
		# inverse has to undo the actual projective divide, not just a linear map.
		matrix = (2.0, 0.3, 5.0, 0.1, 1.5, -2.0, 0.001, 0.0005, 1.0)
		function = HomographyFunction(matrix)
		original = Point2D(37.0, -12.0)
		recovered = function.reverse(function.apply(original))
		assert recovered.x == pytest.approx(original.x, abs=1e-6)
		assert recovered.y == pytest.approx(original.y, abs=1e-6)

	def test_rejects_a_matrix_without_nine_entries(self) -> None:
		with pytest.raises(ValueError, match="9 entries"):
			HomographyFunction((1.0, 0.0, 0.0))


class TestCodecs:
	def test_identity_round_trips(self) -> None:
		codec = IdentityCodec()
		assert codec.decode(codec.encode(IdentityFunction())) == IdentityFunction()

	def test_scale_round_trips(self) -> None:
		codec = ScaleCodec()
		function = ScaleFunction(0.05)
		assert codec.decode(codec.encode(function)) == function

	def test_scale_decode_raises_on_malformed_input(self) -> None:
		with pytest.raises(ValueError, match="malformed scale"):
			ScaleCodec().decode({})

	def test_similarity_round_trips(self) -> None:
		codec = SimilarityCodec()
		function = SimilarityFunction(Transform2D(a=1.0, b=0.2, tx=3.0, c=-0.2, d=1.0, ty=-1.0))
		assert codec.decode(codec.encode(function)) == function

	def test_homography_round_trips(self) -> None:
		codec = HomographyCodec()
		function = HomographyFunction(_scale_matrix(2.0))
		assert codec.decode(codec.encode(function)) == function


class TestRowEncodeDecode:
	def test_round_trips_a_scale_row(self) -> None:
		row = TransformRow(3, _WIDE_ZONE, ScaleFunction(0.05))
		assert decode_row(encode_row(row)) == row

	def test_round_trips_a_homography_row(self) -> None:
		row = TransformRow(
			3, _square(0.0, 0.0, 100.0, 100.0), HomographyFunction(_scale_matrix(2.0))
		)
		assert decode_row(encode_row(row)) == row

	def test_unknown_type_raises(self) -> None:
		with pytest.raises(ValueError, match="unknown transform type"):
			decode_row({"frame_index": 0, "type": "nonsense", "zone": [[0, 0]], "function": {}})

	def test_zone_with_too_few_vertices_raises(self) -> None:
		with pytest.raises(ValueError, match="at least 3"):
			decode_row(
				{
					"frame_index": 0,
					"type": "scale",
					"zone": [[0, 0], [1, 1]],
					"function": {"factor": 1.0},
				}
			)


class TestTransformTableEmpty:
	"""An entirely empty table (e.g. the ego-motion stage of a static-camera run)
	behaves as the identity for every query -- there is no stage to apply."""

	def test_apply_is_a_no_op(self) -> None:
		table = TransformTable([])
		assert table.apply(Point2D(3.0, 4.0), frame_index=11) == Point2D(3.0, 4.0)

	def test_reverse_is_a_no_op(self) -> None:
		table = TransformTable([])
		assert table.reverse(Point2D(3.0, 4.0), frame_index=11) == Point2D(3.0, 4.0)

	def test_is_trivially_invertible(self) -> None:
		assert TransformTable([]).is_invertible


class TestTransformTableExactFrame:
	"""The ego-motion shape: one row per frame, zone = whole canvas."""

	def test_applies_the_exact_frames_function(self) -> None:
		t = Transform2D(a=2.0, b=0.0, tx=1.0, c=0.0, d=2.0, ty=3.0)
		table = TransformTable([TransformRow(5, _WIDE_ZONE, SimilarityFunction(t))])
		assert table.apply(Point2D(1.0, 1.0), frame_index=5) == t.apply(Point2D(1.0, 1.0))

	def test_missing_frame_raises(self) -> None:
		table = TransformTable([TransformRow(0, _WIDE_ZONE, IdentityFunction())])
		with pytest.raises(KeyError, match="frame 5"):
			table.apply(Point2D(0.0, 0.0), frame_index=5)

	def test_reverse_round_trips(self) -> None:
		t = Transform2D(a=2.0, b=0.3, tx=5.0, c=-0.1, d=1.5, ty=-2.0)
		table = TransformTable([TransformRow(0, _WIDE_ZONE, SimilarityFunction(t))])
		original = Point2D(37.0, -12.0)
		forward = table.apply(original, frame_index=0)
		recovered = table.reverse(forward, frame_index=0)
		assert recovered.x == pytest.approx(original.x, abs=1e-9)
		assert recovered.y == pytest.approx(original.y, abs=1e-9)

	def test_function_at_returns_none_for_a_missing_frame(self) -> None:
		table = TransformTable([TransformRow(0, _WIDE_ZONE, IdentityFunction())])
		assert table.function_at(5) is None
		assert table.function_at(0) == IdentityFunction()


class TestTransformTableZoneSelection:
	"""The per-plane homography shape: several rows sharing a frame_index, told apart
	by which zone contains the query point (no notion of 'nearest' -- that's resolved
	once, at write time, into a flat per-frame row, not a table-level concern)."""

	_GROUND = _square(0.0, 0.0, 100.0, 100.0)
	_BRIDGE = _square(200.0, 200.0, 300.0, 300.0)

	def test_uses_the_matching_zones_function(self) -> None:
		table = TransformTable(
			[
				TransformRow(0, self._GROUND, HomographyFunction(_scale_matrix(1.0))),
				TransformRow(0, self._BRIDGE, HomographyFunction(_scale_matrix(10.0))),
			]
		)
		assert table.apply(Point2D(50.0, 50.0), frame_index=0) == Point2D(50.0, 50.0)
		assert table.apply(Point2D(250.0, 250.0), frame_index=0) == Point2D(2500.0, 2500.0)

	def test_point_outside_every_zone_raises(self) -> None:
		table = TransformTable(
			[TransformRow(0, self._GROUND, HomographyFunction(_scale_matrix(1.0)))]
		)
		with pytest.raises(ValueError, match="matches no zone"):
			table.apply(Point2D(-50.0, -50.0), frame_index=0)

	def test_overlapping_zones_raise_ambiguity(self) -> None:
		overlapping_a = _square(0.0, 0.0, 100.0, 100.0)
		overlapping_b = _square(50.0, 50.0, 150.0, 150.0)
		table = TransformTable(
			[
				TransformRow(0, overlapping_a, HomographyFunction(_scale_matrix(1.0))),
				TransformRow(0, overlapping_b, HomographyFunction(_scale_matrix(2.0))),
			]
		)
		with pytest.raises(ValueError, match="matches 2 zones"):
			table.apply(Point2D(75.0, 75.0), frame_index=0)

	def test_multi_zone_reverse_is_ambiguous(self) -> None:
		table = TransformTable(
			[
				TransformRow(0, self._GROUND, HomographyFunction(_scale_matrix(1.0))),
				TransformRow(0, self._BRIDGE, HomographyFunction(_scale_matrix(10.0))),
			]
		)
		with pytest.raises(ValueError, match="ambiguous"):
			table.reverse(Point2D(50.0, 50.0), frame_index=0)

	def test_multi_zone_table_is_not_invertible(self) -> None:
		table = TransformTable(
			[
				TransformRow(0, self._GROUND, HomographyFunction(_scale_matrix(1.0))),
				TransformRow(0, self._BRIDGE, HomographyFunction(_scale_matrix(10.0))),
			]
		)
		assert not table.is_invertible


class TestTransformTableIntrospection:
	def test_kinds_reports_the_distinct_function_types_present(self) -> None:
		table = TransformTable(
			[
				TransformRow(0, _WIDE_ZONE, ScaleFunction(0.05)),
				TransformRow(1, _WIDE_ZONE, ScaleFunction(0.05)),
			]
		)
		assert table.kinds() == {ScaleFunction}

	def test_any_function_returns_a_representative_instance(self) -> None:
		table = TransformTable([TransformRow(0, _WIDE_ZONE, ScaleFunction(0.05))])
		assert table.any_function() == ScaleFunction(0.05)

	def test_any_function_is_none_for_an_empty_table(self) -> None:
		assert TransformTable([]).any_function() is None

	def test_is_invertible_true_when_zones_differ_only_across_frames(self) -> None:
		ground = _square(0.0, 0.0, 100.0, 100.0)
		bridge = _square(200.0, 200.0, 300.0, 300.0)
		# Different zones at different frames, but exactly one row per frame_index --
		# no ambiguity, since invertibility only cares whether reversing a *given*
		# frame's query needs to disambiguate between several rows.
		table = TransformTable(
			[
				TransformRow(0, ground, HomographyFunction(_scale_matrix(1.0))),
				TransformRow(1, bridge, HomographyFunction(_scale_matrix(1.0))),
			]
		)
		assert table.is_invertible


class TestComposedTransform:
	def test_apply_folds_stages_in_order(self) -> None:
		stage1 = TransformTable([TransformRow(0, _WIDE_ZONE, ScaleFunction(2.0))])
		stage2 = TransformTable(
			[
				TransformRow(
					0,
					_WIDE_ZONE,
					SimilarityFunction(Transform2D(a=1.0, b=0.0, tx=1.0, c=0.0, d=1.0, ty=0.0)),
				)
			]
		)
		combined = compose(stage1, stage2)
		# scale first: (10, 20) -> (20, 40), then +1 in x -> (21, 40).
		assert combined.apply(Point2D(10.0, 20.0), frame_index=0) == Point2D(21.0, 40.0)

	def test_compose_invertible_reverses_stages_in_reverse_order(self) -> None:
		stage1 = TransformTable([TransformRow(0, _WIDE_ZONE, ScaleFunction(2.0))])
		stage2 = TransformTable(
			[
				TransformRow(
					0,
					_WIDE_ZONE,
					SimilarityFunction(Transform2D(a=1.0, b=0.0, tx=1.0, c=0.0, d=1.0, ty=0.0)),
				)
			]
		)
		combined = compose_invertible(stage1, stage2)
		original = Point2D(10.0, 20.0)
		forward = combined.apply(original, frame_index=0)
		recovered = combined.reverse(forward, frame_index=0)
		assert recovered.x == pytest.approx(original.x)
		assert recovered.y == pytest.approx(original.y)

	def test_empty_composition_is_the_identity(self) -> None:
		combined = compose()
		assert combined.apply(Point2D(3.0, 4.0), frame_index=0) == Point2D(3.0, 4.0)


class TestTranslationTransform:
	def test_apply_shifts_by_a_constant(self) -> None:
		shift = TranslationTransform(5.0, -3.0)
		assert shift.apply(Point2D(1.0, 1.0), frame_index=0) == Point2D(6.0, -2.0)

	def test_reverse_undoes_the_shift(self) -> None:
		shift = TranslationTransform(5.0, -3.0)
		original = Point2D(1.0, 1.0)
		assert shift.reverse(shift.apply(original, frame_index=0), frame_index=0) == original

	def test_ignores_frame_index(self) -> None:
		shift = TranslationTransform(1.0, 1.0)
		assert shift.apply(Point2D(0.0, 0.0), 0) == shift.apply(Point2D(0.0, 0.0), 999)


class TestLocalScaleAt:
	def test_uniform_scale_recovers_the_scale_factor(self) -> None:
		table = TransformTable(
			[TransformRow(0, _WIDE_ZONE, HomographyFunction(_scale_matrix(0.25)))]
		)
		assert local_scale_at(table, Point2D(100.0, 100.0)) == pytest.approx(0.25)

	def test_empty_table_has_unit_scale(self) -> None:
		assert local_scale_at(TransformTable([]), Point2D(50.0, 50.0)) == pytest.approx(1.0)

	def test_scale_is_local_under_perspective(self) -> None:
		# Foreshortening grows with y; the metres-per-pixel must differ between two rows.
		matrix = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.01, 1.0)
		table = TransformTable([TransformRow(0, _WIDE_ZONE, HomographyFunction(matrix))])
		near = local_scale_at(table, Point2D(0.0, 0.0))
		far = local_scale_at(table, Point2D(0.0, 80.0))
		assert not math.isclose(near, far)
