#!/usr/bin/python

import sys
from collections import OrderedDict as odict
from PySide.QtGui import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QGroupBox
from PySide.QtGui import QComboBox, QPushButton, QTextEdit
from PySide.QtCore import Slot

from cobaya.yaml import yaml_dump
import input_database
from create_input import create_input

_separator = " -- "


def text(key, contents):
    desc = (contents or {}).get(input_database._desc)
    return "%s"%key + (_separator+str(desc) if desc else "")


class MainWindow(QWidget):

    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Cobaya input generator for Cosmology")
        self.setGeometry(0, 0, 1500, 1000)
        self.move(
            QApplication.desktop().screen().rect().center() - self.rect().center())
        self.show()
        # Main layout
        self.layout = QHBoxLayout()
        self.setLayout(self.layout)
        self.layout_options = QVBoxLayout()
        self.layout_output = QVBoxLayout()
        self.layout.addLayout(self.layout_options)
        self.layout.addLayout(self.layout_output)
        # LEFT: Options
        self.atoms = odict()
        titles = odict([
            ["preset", "Presets"],
            ["theory", "Theory code"],
            ["primordial", "Primordial perturbations"],
            ["geometry", "Geometry"],
            ["hubble", "Constaint on hubble parameter"],
            ["baryons", "Baryon sector"],
            ["dark_matter", "Dark matter"],
            ["dark_energy", "Lambda / Dark energy"],
            ["neutrinos", "Neutrinos and other extra matter"],
            ["bbn", "BBN"],
            ["reionization", "Reionization history"],
            ["cmb_lensing", "CMB lensing"],
            ["cmb", "CMB experiments"],
            ["sampler", "Samplers"]])
        for a in titles:
            self.atoms[a] = {
                "group": QGroupBox(titles[a]),
                "combo": QComboBox()}
            self.layout_options.addWidget(self.atoms[a]["group"])
            self.atoms[a]["layout"] = QVBoxLayout(self.atoms[a]["group"])
            self.atoms[a]["layout"].addWidget(self.atoms[a]["combo"])
            self.atoms[a]["combo"].addItems(
                [text(k,v) for k,v in getattr(input_database, a).items()])
        # Connect to refreshers -- needs to be after adding all elements
        for a in self.atoms:
            if a == "preset":
                self.atoms["preset"]["combo"].currentIndexChanged.connect(
                    self.refresh_preset)
                continue
            self.atoms[a]["combo"].currentIndexChanged.connect(self.refresh)
        # Buttons
        self.buttons = QHBoxLayout()
        self.buttons.addStretch(1)
        self.save_button = QPushButton('Save', self)
        self.copy_button = QPushButton('Copy', self)
        self.buttons.addWidget(self.save_button)
        self.buttons.addWidget(self.copy_button)
        # Put buttonts on the bottom-right
        self.layout_options.addStretch(1)
        self.layout_options.addLayout(self.buttons)
        # RIGHT: Output
        self.display = QTextEdit()
        self.display.setLineWrapMode(QTextEdit.NoWrap)
        self.display.setFontFamily("mono")
        self.display.setCursorWidth(0)
        self.display.setReadOnly(True)
        self.layout_output.addWidget(self.display)

    @Slot()
    def refresh(self):
        info = create_input(**{
            k:self.atoms[k]["combo"].currentText().split(_separator)[0]
            for k in self.atoms if k is not "preset"})
        self.refresh_display(info)

    @Slot()
    def refresh_preset(self):
        preset = self.atoms["preset"]["combo"].currentText().split(_separator)[0]
        info = create_input(preset=preset)
        self.refresh_display(info)
        # Update combo boxes to reflect the preset values
        for k,v in input_database.preset[preset].items():
            if k == input_database._desc:
                continue
            self.atoms[k]["combo"].setCurrentIndex(
                self.atoms[k]["combo"].findText(
                    text(v,getattr(input_database, k).get(v))))

    def refresh_display(self, info):
        self.display.setText(yaml_dump(info))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
