
import argparse
import configparser
import logging
from pathlib import Path

from enaml.application import deferred_call
from enaml.qt.QtCore import QStandardPaths


def config_file():
    config_path = Path(QStandardPaths.standardLocations(QStandardPaths.AppConfigLocation)[0])
    config_file =  config_path / 'synaptogram' / 'config.ini'
    config_file.parent.mkdir(exist_ok=True, parents=True)
    return config_file


def get_config():
    config = configparser.ConfigParser()
    config['DEFAULT'] = {'current_path': ''}
    config.read(config_file())
    return config


def write_config(config):
    with config_file().open('w') as fh:
        config.write(fh)


def main():
    import enaml
    from enaml.qt.qt_application import QtApplication
    logging.basicConfig(level='INFO')

    from synaptogram.presenter import SynaptogramPresenter
    from synaptogram.reader import ImarisReader
    with enaml.imports():
        from synaptogram.gui import load_dataset, SynaptogramWindow

    parser = argparse.ArgumentParser("Synaptogram helper")
    parser.add_argument("path", nargs='?')
    args = parser.parse_args()

    app = QtApplication()
    config = get_config()

    view = SynaptogramWindow(
            current_path=config['DEFAULT']['current_path'],
    )
    if args.path is not None:
        deferred_call(load_dataset, args.path, view)

    view.show()
    app.start()
    app.stop()
    config['DEFAULT']['current_path'] = str(Path(view.current_path).absolute())
    write_config(config)


if __name__ == "__main__":
    main()
