"""Tests for glas.gui.widgets.experiment_metadata_widget."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication

from glas.experiment import PhysicalParameters
from glas.gui.widgets.experiment_metadata_widget import (
    ExperimentMetadataWidget,
    _none_if_zero,
)


class TestNoneIfZero:
    def test_zero_becomes_none(self) -> None:
        assert _none_if_zero(0.0) is None

    def test_nonzero_passes_through(self) -> None:
        assert _none_if_zero(2.5) == 2.5


@pytest.fixture
def qapp() -> QApplication:
    app = QApplication.instance()
    return app if app is not None else QApplication([])


@pytest.fixture
def widget(qapp: QApplication) -> ExperimentMetadataWidget:
    return ExperimentMetadataWidget()


class TestDefaultState:
    def test_parameters_are_all_default(self, widget: ExperimentMetadataWidget) -> None:
        assert widget.parameters() == PhysicalParameters()

    def test_datetime_label_is_populated(self, widget: ExperimentMetadataWidget) -> None:
        assert widget._datetime_label.text() != ""


class TestFillingInFields:
    def test_text_fields_are_reflected_in_parameters(
        self, widget: ExperimentMetadataWidget
    ) -> None:
        widget._experiment_id_edit.setText("EXP-1")
        widget._operator_edit.setText("Bijay")
        widget._material_edit.setText("glass beads")
        widget._container_geometry_edit.setText("cylindrical, 80mm ID")

        params = widget.parameters()
        assert params.experiment_id == "EXP-1"
        assert params.operator == "Bijay"
        assert params.material == "glass beads"
        assert params.container_geometry == "cylindrical, 80mm ID"

    def test_numeric_fields_are_reflected_in_parameters(
        self, widget: ExperimentMetadataWidget
    ) -> None:
        widget._grain_diameter_spin.setValue(2.5)
        widget._grain_density_spin.setValue(2500.0)
        widget._fill_depth_spin.setValue(40.0)
        widget._frequency_spin.setValue(60.0)
        widget._amplitude_spin.setValue(1.5)
        widget._target_acceleration_spin.setValue(2.0)

        params = widget.parameters()
        assert params.grain_diameter_mm == pytest.approx(2.5)
        assert params.grain_density_kg_m3 == pytest.approx(2500.0)
        assert params.fill_depth_mm == pytest.approx(40.0)
        assert params.frequency_hz == pytest.approx(60.0)
        assert params.amplitude_mm == pytest.approx(1.5)
        assert params.target_acceleration_g == pytest.approx(2.0)

    def test_untouched_numeric_fields_stay_none(self, widget: ExperimentMetadataWidget) -> None:
        widget._grain_diameter_spin.setValue(2.5)
        params = widget.parameters()
        assert params.grain_density_kg_m3 is None
        assert params.fill_depth_mm is None


class TestSetParameters:
    def test_round_trips_through_the_form(self, widget: ExperimentMetadataWidget) -> None:
        original = PhysicalParameters(
            experiment_id="EXP-42",
            operator="bijay",
            material="sand",
            grain_diameter_mm=0.5,
            grain_density_kg_m3=2650.0,
            container_geometry="cylindrical",
            fill_depth_mm=40.0,
            frequency_hz=60.0,
            amplitude_mm=1.5,
            target_acceleration_g=2.0,
        )
        widget.set_parameters(original)
        assert widget.parameters() == original

    def test_setting_default_parameters_clears_the_form(
        self, widget: ExperimentMetadataWidget
    ) -> None:
        widget.set_parameters(PhysicalParameters(experiment_id="EXP-1", grain_diameter_mm=2.0))
        widget.set_parameters(PhysicalParameters())
        assert widget.parameters() == PhysicalParameters()
