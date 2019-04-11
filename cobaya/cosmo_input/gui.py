# Python 2/3 compatibility
from __future__ import absolute_import, division

# Global
import os
import sys
import signal
from collections import OrderedDict as odict
from pprint import pformat

# Local
from cobaya.yaml import yaml_dump
from cobaya.cosmo_input import input_database
from cobaya.cosmo_input.create_input import create_input
from cobaya.citation import prettyprint_citation, citation
from cobaya.tools import warn_deprecation

try:
    from PySide.QtGui import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QGroupBox
    from PySide.QtGui import QScrollArea, QTabWidget, QComboBox, QPushButton, QTextEdit
    from PySide.QtGui import QFileDialog, QCheckBox, QLabel
    from PySide.QtCore import Slot
except ImportError:
    try:
        from PySide2.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QGroupBox
        from PySide2.QtWidgets import QScrollArea, QTabWidget, QComboBox, QPushButton, QTextEdit
        from PySide2.QtWidgets import QFileDialog, QCheckBox, QLabel
        from PySide2.QtCore import Slot
    except ImportError:
        QWidget, Slot = object, (lambda: lambda *x: None)

# Quit with C-c
signal.signal(signal.SIGINT, signal.SIG_DFL)


def text(key, contents):
    desc = (contents or {}).get(input_database._desc)
    return desc or key


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Cobaya input generator for Cosmology")
        self.setGeometry(0, 0, 1500, 1000)
        self.move(
            QApplication.desktop().screenGeometry().center() - self.rect().center())
        self.show()
        # Main layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.layout_left = QVBoxLayout()
        self.layout.addLayout(self.layout_left)
        self.layout_output = QVBoxLayout()
        self.layout.addLayout(self.layout_output)
        # LEFT: Options
        self.options = QWidget()
        self.layout_options = QVBoxLayout()
        self.options.setLayout(self.layout_options)
        self.options_scroll = QScrollArea()
        self.options_scroll.setWidget(self.options)
        self.options_scroll.setWidgetResizable(True)
        self.layout_left.addWidget(self.options_scroll)
        titles = odict([
            ["Presets", odict([["preset", "Presets"]])],
            ["Cosmological Model", odict([
                ["theory", "Theory code"],
                ["primordial", "Primordial perturbations"],
                ["geometry", "Geometry"],
                ["hubble", "Hubble parameter constraint"],
                ["matter", "Matter sector"],
                ["neutrinos", "Neutrinos and other extra matter"],
                ["dark_energy", "Lambda / Dark energy"],
                ["bbn", "BBN"],
                ["reionization", "Reionization history"]])],
            ["Data sets", odict([
                ["like_cmb", "CMB experiments"],
                ["like_bao", "BAO experiments"],
                ["like_sn", "SN experiments"],
                ["like_H0", "Local H0 measurements"]])],
            ["Sampler", odict([["sampler", "Samplers"]])]])
        self.combos = odict()
        for group, fields in titles.items():
            group_box = QGroupBox(group)
            self.layout_options.addWidget(group_box)
            group_layout = QVBoxLayout(group_box)
            for a, desc in fields.items():
                self.combos[a] = QComboBox()
                if len(fields) > 1:
                    label = QLabel(desc)
                    group_layout.addWidget(label)
                group_layout.addWidget(self.combos[a])
                self.combos[a].addItems(
                    [text(k, v) for k, v in getattr(input_database, a).items()])
        # PLANCK NAMES CHECKBOX TEMPORARILY DISABLED
        #                if a == "theory":
        #                    # Add Planck-naming checkbox
        #                    self.planck_names = QCheckBox(
        #                        "Keep common parameter names "
        #                        "(useful for fast CLASS/CAMB switching)")
        #                    group_layout.addWidget(self.planck_names)
        # Connect to refreshers -- needs to be after adding all elements
        for field, combo in self.combos.items():
            if field == "preset":
                combo.currentIndexChanged.connect(self.refresh_preset)
            else:
                combo.currentIndexChanged.connect(self.refresh)
        #        self.planck_names.stateChanged.connect(self.refresh_keep_preset)
        # RIGHT: Output + buttons
        self.display_tabs = QTabWidget()
        self.display = {}
        for k in ["yaml", "python", "citations"]:
            self.display[k] = QTextEdit()
            self.display[k].setLineWrapMode(QTextEdit.NoWrap)
            self.display[k].setFontFamily("mono")
            self.display[k].setCursorWidth(0)
            self.display[k].setReadOnly(True)
            self.display_tabs.addTab(self.display[k], k)
        self.layout_output.addWidget(self.display_tabs)
        # Buttons
        self.buttons = QHBoxLayout()
        self.save_button = QPushButton('Save', self)
        self.copy_button = QPushButton('Copy to clipboard', self)
        self.buttons.addWidget(self.save_button)
        self.buttons.addWidget(self.copy_button)
        self.save_button.released.connect(self.save_file)
        self.copy_button.released.connect(self.copy_clipb)
        self.layout_output.addLayout(self.buttons)
        self.save_dialog = QFileDialog()
        self.save_dialog.setFileMode(QFileDialog.AnyFile)
        self.save_dialog.setAcceptMode(QFileDialog.AcceptSave)

    def create_input(self):
        return create_input(
            get_comments=True,
            #           planck_names=self.planck_names.isChecked(),
            **{field: list(getattr(input_database, field).keys())[combo.currentIndex()]
               for field, combo in self.combos.items() if field is not "preset"})

    @Slot()
    def refresh_keep_preset(self):
        self.refresh_display(self.create_input())

    @Slot()
    def refresh(self):
        self.combos["preset"].blockSignals(True)
        self.combos["preset"].setCurrentIndex(0)
        self.combos["preset"].blockSignals(False)
        self.refresh_display(self.create_input())

    @Slot()
    def refresh_preset(self):
        preset = list(getattr(input_database, "preset").keys())[
            self.combos["preset"].currentIndex()]
        info = create_input(
            get_comments=True,
            #            planck_names=self.planck_names.isChecked(),
            preset=preset)
        self.refresh_display(info)
        # Update combo boxes to reflect the preset values, without triggering update
        for k, v in input_database.preset[preset].items():
            if k in [input_database._desc]:
                continue
            self.combos[k].blockSignals(True)
            self.combos[k].setCurrentIndex(
                self.combos[k].findText(
                    text(v, getattr(input_database, k).get(v))))
            self.combos[k].blockSignals(False)

    def refresh_display(self, info):
        try:
            comments = info.pop(input_database._comment, None)
            comments_text = "\n# " + "\n# ".join(comments)
        except (TypeError,  # No comments
                AttributeError):  # Failed to generate info (returned str instead)
            comments_text = ""
        self.display["python"].setText(
            "from collections import OrderedDict\n\ninfo = " +
            pformat(info) + comments_text)
        self.display["yaml"].setText(yaml_dump(info) + comments_text)
        self.display["citations"].setText(prettyprint_citation(citation(info)))

    @Slot()
    def save_file(self):
        ftype = next(k for k, w in self.display.items()
                     if w is self.display_tabs.currentWidget())
        ffilter = {"yaml": "Yaml files (*.yaml *.yml)", "python": "(*.py)",
                   "citations": "(*.txt)"}[ftype]
        fsuffix = {"yaml": ".yaml", "python": ".py", "citations": ".txt"}[ftype]
        fname, path = self.save_dialog.getSaveFileName(
            self.save_dialog, "Save input file", fsuffix, ffilter, os.getcwd())
        if not fname.endswith(fsuffix):
            fname += fsuffix
        with open(fname, "w+") as f:
            f.write(self.display_tabs.currentWidget().toPlainText())

    @Slot()
    def copy_clipb(self):
        self.clipboard.setText(self.display_tabs.currentWidget().toPlainText())


def gui_script():
    warn_deprecation()
    try:
        app = QApplication(sys.argv)
    except NameError:
        # TODO: fix this long logger setup
        from cobaya.log import logger_setup, HandledException
        logger_setup(0, None)
        import logging
        logging.getLogger("cosmo_generator").error(
            "PySide or PySide2 is not installed! "
            "Check Cobaya's documentation for the cosmo_generator "
            "('Basic cosmology runs').")
        raise HandledException
    clip = app.clipboard()
    window = MainWindow()
    window.clipboard = clip
    sys.exit(app.exec_())


if __name__ == '__main__':
    gui_script()
