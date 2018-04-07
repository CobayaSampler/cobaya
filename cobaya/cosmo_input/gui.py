#!/usr/bin/python

import sys
from collections import OrderedDict as odict
from PySide.QtGui import QWidget, QApplication, QVBoxLayout, QHBoxLayout, QGroupBox
from PySide.QtGui import QComboBox, QPushButton, QTextEdit
from PySide.QtCore import Slot

from input_database import _desc, theory, primordial, reionization
from create_input import create_input

_separator = " -- "


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
        # Presets
        self.group_presets = QGroupBox("Presets")
        self.layout_options.addWidget(self.group_presets)
        self.layout_presets = QVBoxLayout(self.group_presets)
        self.presets_combo = QComboBox()
        self.layout_presets.addWidget(self.presets_combo)
        self.atoms = odict()
        titles = {"theory": "Theory code",
                  "primordial": "Primordial perturbations",
                  "reionization": "Reionization history"}
        for a in ["theory", "primordial", "reionization"]:
            self.atoms[a] = {
                "group": QGroupBox(titles[a]),
                "combo": QComboBox()}
            self.layout_options.addWidget(self.atoms[a]["group"])
            self.atoms[a]["layout"] = QVBoxLayout(self.atoms[a]["group"])
            self.atoms[a]["layout"].addWidget(self.atoms[a]["combo"])
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
#        # EXAMPLE TEXT #######################
#        with open("../tests/test_cosmo_grid.yaml", "r") as f:
#            text = "".join(f.readlines())
#        self.display.setText(text)
        # POPULATING ################################################
        self.atoms["theory"]["combo"].addItems(list(theory.keys()))
        self.atoms["primordial"]["combo"].addItems(
            ["%s%s%s"%(k,_separator,v[_desc]) for k,v in primordial.items()])
        self.atoms["reionization"]["combo"].addItems(list(reionization.keys()))

    @Slot()
    def refresh(self):
        info = create_input(
            str(self.atoms["theory"]["combo"].currentText().split(_separator)[0]),
            self.atoms["primordial"]["combo"].currentText().split(_separator)[0],
            self.atoms["reionization"]["combo"].currentText().split(_separator)[0])
        from cobaya.yaml import yaml_dump
        self.display.setText(yaml_dump(info))


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
