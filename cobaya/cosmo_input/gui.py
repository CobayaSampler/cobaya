import io
import os
import platform
import signal
import sys
from pprint import pformat

import numpy as np

from cobaya.bib import get_bib_component, get_bib_info, pretty_repr_bib
from cobaya.conventions import kinds, packages_path_env, packages_path_input, subfolders
from cobaya.cosmo_input import input_database
from cobaya.cosmo_input.autoselect_covmat import covmat_folders, get_best_covmat
from cobaya.cosmo_input.create_input import create_input
from cobaya.cosmo_input.input_database import _combo_dict_text
from cobaya.input import get_default_info
from cobaya.tools import (
    cov_to_std_and_corr,
    get_available_internal_class_names,
    resolve_packages_path,
    sort_cosmetic,
    warn_deprecation,
)
from cobaya.yaml import yaml_dump

# per-platform settings for correct high-DPI scaling
if platform.system() == "Linux":
    font_size = "12px"
else:  # Windows/Mac
    font_size = "9pt"

try:
    from PySide6.QtCore import QPoint, QSettings, QSize, Qt, Slot
    from PySide6.QtGui import QAction, QColor
    from PySide6.QtWidgets import (
        QAbstractItemView,
        QApplication,
        QComboBox,
        QFileDialog,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMenuBar,
        QPushButton,
        QScrollArea,
        QTableWidget,
        QTableWidgetItem,
        QTabWidget,
        QTextEdit,
        QVBoxLayout,
        QWidget,
    )

except ImportError:
    QWidget, Slot = object, (lambda: lambda *x: None)


def text(key, contents):
    desc = (contents or {}).get("desc")
    return desc or key


def get_settings():
    return QSettings("cobaya", "gui")


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cobaya input generator for Cosmology")
        self.setStyleSheet("* {font-size:%s;}" % font_size)
        # Menu bar for defaults
        self.menubar = QMenuBar()
        defaults_menu = self.menubar.addMenu(
            "&Show defaults and bibliography for a component..."
        )
        menu_actions = {}
        for kind in kinds:
            submenu = defaults_menu.addMenu(subfolders[kind])
            components = get_available_internal_class_names(kind)
            menu_actions[kind] = {}
            for component in components:
                menu_actions[kind][component] = QAction(component, self)
                menu_actions[kind][component].setData((kind, component))
                menu_actions[kind][component].triggered.connect(self.show_defaults)
                submenu.addAction(menu_actions[kind][component])
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
        self.combos = dict()
        for group, fields in _combo_dict_text:
            group_box = QGroupBox(group)
            self.layout_options.addWidget(group_box)
            group_layout = QVBoxLayout(group_box)
            for a, desc in fields:
                self.combos[a] = QComboBox()
                # Combo box label only if not single element in group
                if len(fields) > 1:
                    label = QLabel(desc)
                    group_layout.addWidget(label)
                group_layout.addWidget(self.combos[a])
                self.combos[a].addItems(
                    [text(k, v) for k, v in getattr(input_database, a).items()]
                )
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
        self.display["covmat"] = QWidget()
        covmat_tab_layout = QVBoxLayout()
        self.display["covmat"].setLayout(covmat_tab_layout)
        self.covmat_text = QLabel()
        self.covmat_text.setWordWrap(True)
        self.covmat_table = QTableWidget(0, 0)
        self.covmat_table.setEditTriggers(QAbstractItemView.NoEditTriggers)  # ReadOnly!
        covmat_tab_layout.addWidget(self.covmat_text)
        covmat_tab_layout.addWidget(self.covmat_table)
        self.display_tabs.addTab(self.display["covmat"], "covariance matrix")
        self.layout_output.addWidget(self.display_tabs)
        # Buttons
        self.buttons = QHBoxLayout()
        self.save_button = QPushButton("Save as...", self)
        self.copy_button = QPushButton("Copy to clipboard", self)
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

    def getScreen(self):
        try:
            return self.screen().availableGeometry()
        except Exception:
            return QApplication.screenAt(
                self.mapToGlobal(QPoint(self.width() // 2, 0))
            ).availableGeometry()

    def read_settings(self):
        settings = get_settings()
        screen = self.getScreen()
        h = min(screen.height() * 5 / 6.0, 900)
        size = QSize(min(screen.width() * 5 / 6.0, 1200), h)
        pos = settings.value("pos", None)
        savesize = settings.value("size", size)
        if savesize.width() > screen.width():
            savesize.setWidth(size.width())
        if savesize.height() > screen.height():
            savesize.setHeight(size.height())
        self.resize(savesize)
        if (
            pos is None
            or pos.x() + savesize.width() > screen.width()
            or pos.y() + savesize.height() > screen.height()
        ):
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
            **{
                field: list(getattr(input_database, field))[combo.currentIndex()]
                for field, combo in self.combos.items()
                if field != "preset"
            },
        )

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
            self.combos["preset"].currentIndex()
        ]
        if preset is input_database.none:
            return
        info = create_input(
            get_comments=True,
            #            planck_names=self.planck_names.isChecked(),
            preset=preset,
        )
        self.refresh_display(info)
        # Update combo boxes to reflect the preset values, without triggering update
        for k, v in input_database.preset[preset].items():
            if k in ["desc"]:
                continue
            self.combos[k].blockSignals(True)
            self.combos[k].setCurrentIndex(
                self.combos[k].findText(text(v, getattr(input_database, k).get(v)))
            )
            self.combos[k].blockSignals(False)

    def refresh_display(self, info):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            comments = info.pop("comment", None)
            comments_text = "\n# " + "\n# ".join(comments)
        except (
            TypeError,  # No comments
            AttributeError,
        ):  # Failed to generate info (returned str instead)
            comments_text = ""
        self.display["python"].setText("info = " + pformat(info) + comments_text)
        self.display["yaml"].setText(
            (info if isinstance(info, str) else yaml_dump(sort_cosmetic(info)))
            + comments_text
        )
        self.display["bibliography"].setText(pretty_repr_bib(*get_bib_info(info)))
        # Display covmat
        packages_path = resolve_packages_path()
        if not packages_path:
            self.covmat_text.setText(
                "\nIn order to find a covariance matrix, you need to define an external "
                "packages installation path, e.g. via the env variable %r.\n"
                % packages_path_env
            )
        elif all(
            not os.path.isdir(d.format(**{packages_path_input: packages_path}))
            for d in covmat_folders
        ):
            self.covmat_text.setText(
                "\nThe external cosmological packages appear not to be installed where "
                "expected: %r\n" % packages_path
            )
        else:
            covmat_data = get_best_covmat(info, packages_path=packages_path)
            self.current_params_in_covmat = covmat_data["params"]
            self.current_covmat = covmat_data["covmat"]
            _, corrmat = cov_to_std_and_corr(self.current_covmat)
            self.covmat_text.setText(
                "\nCovariance file: %r\n\n"
                "NB: you do *not* need to save or copy this covariance matrix: "
                "it will be selected automatically.\n" % covmat_data["name"]
            )
            self.covmat_table.setRowCount(len(self.current_params_in_covmat))
            self.covmat_table.setColumnCount(len(self.current_params_in_covmat))
            self.covmat_table.setHorizontalHeaderLabels(
                list(self.current_params_in_covmat)
            )
            self.covmat_table.setVerticalHeaderLabels(list(self.current_params_in_covmat))
            # Color map for correlations
            from matplotlib import colormaps

            cmap_corr = colormaps["coolwarm_r"]
            for i, pi in enumerate(self.current_params_in_covmat):
                for j, pj in enumerate(self.current_params_in_covmat):
                    self.covmat_table.setItem(
                        i, j, QTableWidgetItem("%g" % self.current_covmat[i, j])
                    )
                    if i != j:
                        color = [256 * c for c in cmap_corr(corrmat[i, j] / 2 + 0.5)[:3]]
                    else:
                        color = [255.99] * 3
                    self.covmat_table.item(i, j).setBackground(QColor(*color))
                    self.covmat_table.item(i, j).setForeground(QColor(0, 0, 0))
        QApplication.restoreOverrideCursor()

    def save_covmat_txt(self, file_handle=None):
        """
        Saved covmat to given file_handle. If none given, returns the text to be saved.
        """
        return_txt = False
        if not file_handle:
            file_handle = io.BytesIO()
            return_txt = True
        np.savetxt(
            file_handle,
            self.current_covmat,
            header=" ".join(self.current_params_in_covmat),
        )
        if return_txt:
            return file_handle.getvalue().decode()

    @Slot()
    def save_file(self):
        ftype = next(
            k for k, w in self.display.items() if w is self.display_tabs.currentWidget()
        )
        ffilter = {
            "yaml": "Yaml files (*.yaml *.yml)",
            "python": "(*.py)",
            "bibliography": "(*.txt)",
            "covmat": "(*.covmat)",
        }[ftype]
        fsuffix = {
            "yaml": ".yaml",
            "python": ".py",
            "bibliography": ".txt",
            "covmat": ".covmat",
        }[ftype]
        fname, path = self.save_dialog.getSaveFileName(
            self.save_dialog, "Save input file", fsuffix, ffilter, os.getcwd()
        )
        if not fname.endswith(fsuffix):
            fname += fsuffix
        with open(fname, "w+", encoding="utf-8") as f:
            if self.display_tabs.currentWidget() == self.display["covmat"]:
                self.save_covmat_txt(f)
            else:
                f.write(self.display_tabs.currentWidget().toPlainText())

    @Slot()
    def copy_clipb(self):
        clipboard = QApplication.clipboard()
        if self.display_tabs.currentWidget() == self.display["covmat"]:
            clipboard.setText(self.save_covmat_txt())
        else:
            clipboard.setText(self.display_tabs.currentWidget().toPlainText())

    def show_defaults(self):
        kind, component = self.sender().data()
        self.current_defaults_diag = DefaultsDialog(kind, component, parent=self)


class DefaultsDialog(QWidget):
    def __init__(self, kind, component, parent=None):
        super().__init__()
        self.setWindowTitle(f"{kind} : {component}")
        self.setGeometry(0, 0, 500, 500)
        self.move(parent.getScreen().center() - self.rect().center())
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
        defaults_txt = get_default_info(component, kind, return_yaml=True)
        _indent = "  "
        defaults_txt = (
            kind
            + ":\n"
            + _indent
            + component
            + ":\n"
            + 2 * _indent
            + ("\n" + 2 * _indent).join(defaults_txt.split("\n"))
        )
        from cobaya.yaml import yaml_load

        self.display["python"].setText(pformat(yaml_load(defaults_txt)))
        self.display["yaml"].setText(defaults_txt)
        self.display["bibliography"].setText(get_bib_component(component, kind))
        # Buttons
        self.buttons = QHBoxLayout()
        self.close_button = QPushButton("Close", self)
        self.copy_button = QPushButton("Copy to clipboard", self)
        self.buttons.addWidget(self.close_button)
        self.buttons.addWidget(self.copy_button)
        self.close_button.released.connect(self.close)
        self.copy_button.released.connect(self.copy_clipb)
        self.layout.addLayout(self.buttons)

    @Slot()
    def copy_clipb(self):
        QApplication.clipboard().setText(self.display_tabs.currentWidget().toPlainText())


def gui_script():
    warn_deprecation()
    try:
        app = QApplication(sys.argv)
    except NameError:
        # TODO: fix this long logger setup
        from cobaya.log import LoggedError, logger_setup

        logger_setup(0, None)
        raise LoggedError(
            "cosmo_generator",
            "PySide is not installed! "
            "Check Cobaya's documentation for the cosmo_generator "
            "('Basic cosmology runs').",
        )

    # Quit with C-c
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    gui_script()
