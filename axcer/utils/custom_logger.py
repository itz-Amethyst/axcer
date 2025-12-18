import logging
import logging.handlers
import re
import sys


class ColorCodes:
    grey = "\x1b[38;21m"
    green = "\x1b[1;32m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[1;34m"
    light_blue = "\x1b[1;36m"
    purple = "\x1b[1;35m"
    reset = "\x1b[0m"


class CustomFormatter(logging.Formatter):
    arg_colors = [ColorCodes.purple, ColorCodes.light_blue]
    level_fields = ["levelname", "levelno"]
    level_to_color = {
        logging.DEBUG: ColorCodes.grey,
        logging.INFO: ColorCodes.green,
        logging.WARNING: ColorCodes.yellow,
        logging.ERROR: ColorCodes.red,
        logging.CRITICAL: ColorCodes.bold_red,
    }

    def __init__(self, fmt: str):
        super().__init__()
        self.level_to_formatter = {}

        def add_color_format(level: int):
            color = CustomFormatter.level_to_color[level]
            _format = fmt
            for fld in CustomFormatter.level_fields:
                search = r"(%\(" + fld + r"\).*?s)"
                _format = re.sub(search, f"{color}\\1{ColorCodes.reset}", _format)
            formatter = logging.Formatter(_format)
            self.level_to_formatter[level] = formatter

        add_color_format(logging.DEBUG)
        add_color_format(logging.INFO)
        add_color_format(logging.WARNING)
        add_color_format(logging.ERROR)
        add_color_format(logging.CRITICAL)

    @staticmethod
    def rewrite_record(record: logging.LogRecord):
        msg = record.msg
        msg = msg.replace("{", "_{{")
        msg = msg.replace("}", "_}}")
        placeholder_count = 0
        while True:
            if "_{{" not in msg:
                break
            color_index = placeholder_count % len(CustomFormatter.arg_colors)
            color = CustomFormatter.arg_colors[color_index]
            msg = msg.replace("_{{", color + "{", 1)
            msg = msg.replace("_}}", "}" + ColorCodes.reset, 1)
            placeholder_count += 1

        record.msg = msg.format(*record.args)
        record.args = []

    def format(self, record):
        orig_msg = record.msg
        orig_args = record.args
        formatter = self.level_to_formatter.get(record.levelno)
        self.rewrite_record(record)
        formatted = formatter.format(record)

        record.msg = orig_msg
        record.args = orig_args
        return formatted


def init_logging():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    console_level = "ERROR"
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(console_level)
    console_format = "%(asctime)s - %(levelname)-8s - %(name)-25s - %(message)s"
    colored_formatter = CustomFormatter(console_format)
    console_handler.setFormatter(colored_formatter)
    root_logger.addHandler(console_handler)

    return root_logger


logger = init_logging()
