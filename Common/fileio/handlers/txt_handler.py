# Copyright (c) OpenMMLab. All rights reserved.


from .base import BaseFileHandler


class TxtHandler(BaseFileHandler):
    def load_from_fileobj(self, file, **kwargs):
        txt_lines = []
        for line in file.readlines():
            line = line.strip()
            txt_lines.append(line)
        return txt_lines

    def dump_to_fileobj(self, obj, file, **kwargs):
        for item in obj:
            s = item + '\n'
            file.write(s)

    def dump_to_str(self, obj, **kwargs):
        pass
        # TODO implementation
