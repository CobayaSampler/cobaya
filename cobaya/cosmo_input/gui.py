# Global
import os
import sys
import platform
import signal
from pprint import pformat

# Local
from cobaya.yaml import yaml_dump
from cobaya.cosmo_input import input_database
from cobaya.cosmo_input.create_input import create_input
from cobaya.bib import prettyprint_bib, get_bib_info, get_bib_module
from cobaya.tools import warn_deprecation, get_available_internal_class_names
from cobaya.input import get_default_info
from cobaya.conventions import subfolders, kinds

# per-platform settings for correct high-DPI scaling
if platform.system() == "Linux":
    font_size = "12px"
    set_attributes = ["AA_EnableHighDpiScaling"]
else:  # Windows/Mac
    font_size = "9pt"
    set_attributes = ["AA_EnableHighDpiScaling"]

try:
    # noinspection PyUnresolvedReferences
    from PySide2.QtWidgets import QWidget, QApplication, QVBoxLayout, QHBoxLayout, \
        QGroupBox, QScrollArea, QTabWidget, QComboBox, QPushButton, QTextEdit, \
        QFileDialog, QCheckBox, QLabel, QMenuBar, QAction, QDialog
    # noinspection PyUnresolvedReferences
    from PySide2.QtCore import Slot, Qt, QCoreApplication, QSize, QSettings

    for attribute in set_attributes:
        QApplication.setAttribute(getattr(Qt, attribute))
except ImportError:
    QWidget, Slot = object, (lambda: lambda *x: None)

# Quit with C-c
signal.signal(signal.SIGINT, signal.SIG_DFL)


def text(key, contents):
    desc = (contents or {}).get(input_database._desc)
    return desc or key


def get_settings():
    return QSettings('cobaya', 'gui')


class MainWindow(QWidget):

    # noinspection PyUnresolvedReferences
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cobaya input generator for Cosmology")
        self.setStyleSheet("* {font-size:%s;}" % font_size)
        # Menu bar for defaults
        self.menubar = QMenuBar()
        defaults_menu = self.menubar.addMenu(
            '&Show defaults and bibliography for a module...')
        menu_actions = {}
        for kind in kinds:
            submenu = defaults_menu.addMenu(subfolders[kind])
            modules = get_available_internal_class_names(kind)
            menu_actions[kind] = {}
            for module in modules:
                menu_actions[kind][module] = QAction(module, self)
                menu_actions[kind][module].setData((kind, module))
                menu_actions[kind][module].triggered.connect(self.show_defaults)
                submenu.addAction(menu_actions[kind][module])
        # Main layout
        self.menu_layout = QVBoxLayout()
        self.menu_layout.addWidget(self.menubar)
        self.setLayout(self.menu_layout)
        self.layout = QHBoxLayout()
        self.menu_layout.addLayout(self.layout)
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
        titles = dict([
            ["Presets", {"preset": "Presets"}],
            ["Cosmological Model", dict([
                ["theory", "Theory code"],
                ["primordial", "Primordial perturbations"],
                ["geometry", "Geometry"],
                ["hubble", "Hubble parameter constraint"],
                ["matter", "Matter sector"],
                ["neutrinos", "Neutrinos and other extra matter"],
                ["dark_energy", "Lambda / Dark energy"],
                ["bbn", "BBN"],
                ["reionization", "Reionization history"]])],
            ["Data sets", dict([
                ["like_cmb", "CMB experiments"],
                ["like_bao", "BAO experiments"],
                ["like_des", "DES measurements"],
                ["like_sn", "SN experiments"],
                ["like_H0", "Local H0 measurements"]])],
            ["Sampler", {"sampler": "Samplers"}]])
        self.combos = dict()
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
        for k in ["yaml", "python", "bibliography"]:
            self.display[k] = QTextEdit()
            self.display[k].setLineWrapMode(QTextEdit.NoWrap)
            self.display[k].setFontFamily("mono")
            self.display[k].setCursorWidth(0)
            self.display[k].setReadOnly(True)
            self.display_tabs.addTab(self.display[k], k)
        self.layout_output.addWidget(self.display_tabs)
        # Buttons
        self.buttons = QHBoxLayout()
        self.save_button = QPushButton('Save as...', self)
        self.copy_button = QPushButton('Copy to clipboard', self)
        self.buttons.addWidget(self.save_button)
        self.buttons.addWidget(self.copy_button)
        self.save_button.released.connect(self.save_file)
        self.copy_button.released.connect(self.copy_clipb)
        self.layout_output.addLayout(self.buttons)
        self.save_dialog = QFileDialog()
        self.save_dialog.setFileMode(QFileDialog.AnyFile)
        self.save_dialog.setAcceptMode(QFileDialog.AcceptSave)
        self.read_settings()
        self.show()

    def read_settings(self):
        settings = get_settings()
        # noinspection PyArgumentList
        screen = QApplication.desktop().screenGeometry()
        h = min(screen.height() * 5 / 6., 900)
        size = QSize(min(screen.width() * 5 / 6., 1200), h)
        pos = settings.value("pos", None)
        savesize = settings.value("size", size)
        if savesize.width() > screen.width():
            savesize.setWidth(size.width())
        if savesize.height() > screen.height():
            savesize.setHeight(size.height())
        self.resize(savesize)
        if ((pos is None or pos.x() + savesize.width() > screen.width() or
             pos.y() + savesize.height() > screen.height())):
            self.move(screen.center() - self.rect().center())
        else:
            self.move(pos)

    def write_settings(self):
        settings = get_settings()
        settings.setValue("pos", self.pos())
        settings.setValue("size", self.size())

    def closeEvent(self, event):
        self.write_settings()
        event.accept()

    def create_input(self):
        return create_input(
            get_comments=True,
            #           planck_names=self.planck_names.isChecked(),
            **{field: list(getattr(input_database, field))[combo.currentIndex()]
               for field, combo in self.combos.items() if field != "preset"})

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
        preset = list(getattr(input_database, "preset"))[
            self.combos["preset"].currentIndex()]
        if preset is input_database._none:
            return
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
        self.display["python"].setText("info = " + pformat(info) + comments_text)
        self.display["yaml"].setText(yaml_dump(info) + comments_text)
        self.display["bibliography"].setText(prettyprint_bib(get_bib_info(info)))

    @Slot()
    def save_file(self):
        ftype = next(k for k, w in self.display.items()
                     if w is self.display_tabs.currentWidget())
        ffilter = {"yaml": "Yaml files (*.yaml *.yml)", "python": "(*.py)",
                   "bibliography": "(*.txt)"}[ftype]
        fsuffix = {"yaml": ".yaml", "python": ".py", "bibliography": ".txt"}[ftype]
        fname, path = self.save_dialog.getSaveFileName(
            self.save_dialog, "Save input file", fsuffix, ffilter, os.getcwd())
        if not fname.endswith(fsuffix):
            fname += fsuffix
        with open(fname, "w+", encoding="utf-8") as f:
            f.write(self.display_tabs.currentWidget().toPlainText())

    @Slot()
    def copy_clipb(self):
        self.clipboard.setText(self.display_tabs.currentWidget().toPlainText())

    def show_defaults(self):
        kind, module = self.sender().data()
        self.current_defaults_diag = DefaultsDialog(kind, module, parent=self)


# noinspection PyUnresolvedReferences
class DefaultsDialog(QWidget):

    def __init__(self, kind, module, parent=None):
        super().__init__()
        self.clipboard = parent.clipboard
        self.setWindowTitle("%s : %s" % (kind, module))
        self.setGeometry(0, 0, 500, 500)
        # noinspection PyArgumentList
        self.move(
            QApplication.desktop().screenGeometry().center() - self.rect().center())
        self.show()
        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.display_tabs = QTabWidget()
        self.display = {}
        for k in ["yaml", "python", "bibliography"]:
            self.display[k] = QTextEdit()
            self.display[k].setLineWrapMode(QTextEdit.NoWrap)
            self.display[k].setFontFamily("mono")
            self.display[k].setCursorWidth(0)
            self.display[k].setReadOnly(True)
            self.display_tabs.addTab(self.display[k], k)
        self.layout.addWidget(self.display_tabs)
        # Fill text
        defaults_txt = get_default_info(module, kind, return_yaml=True)
        _indent = "  "
        defaults_txt = (kind + ":\n" + _indent + module + ":\n" +
                        2 * _indent + ("\n" + 2* _indent).join(defaults_txt.split("\n")))
        from cobaya.yaml import yaml_load
        self.display["python"].setText(pformat(yaml_load(defaults_txt)))
        self.display["yaml"].setText(defaults_txt)
        self.display["bibliography"].setText(get_bib_module(module, kind))
        # Buttons
        self.buttons = QHBoxLayout()
        self.close_button = QPushButton('Close', self)
        self.copy_button = QPushButton('Copy to clipboard', self)
        self.buttons.addWidget(self.close_button)
        self.buttons.addWidget(self.copy_button)
        self.close_button.released.connect(self.close)
        self.copy_button.released.connect(self.copy_clipb)
        self.layout.addLayout(self.buttons)

    @Slot()
    def copy_clipb(self):
        self.clipboard.setText(self.display_tabs.currentWidget().toPlainText())


def gui_script():
    warn_deprecation()
    try:
        app = QApplication(sys.argv)
    except NameError:
        # TODO: fix this long logger setup
        from cobaya.log import logger_setup, LoggedError
        logger_setup(0, None)
        import logging
        raise LoggedError(
            logging.getLogger("cosmo_generator"),
            "PySide2 is not installed! "
            "Check Cobaya's documentation for the cosmo_generator "
            "('Basic cosmology runs').")
    clip = app.clipboard()
    window = MainWindow()
    window.clipboard = clip
    sys.exit(app.exec_())


if __name__ == '__main__':
    gui_script()
