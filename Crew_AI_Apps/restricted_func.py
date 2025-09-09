import os
from pydantic import Field
from crewai_tools import FileWriterTool, FileReadTool

class RestrictedFileWriterTool(FileWriterTool):
    base_dir: str = Field(..., description="Root dir for all writes")

    def run(self, filename: str, content: str):
        root = os.path.abspath(self.base_dir)
        target = os.path.abspath(os.path.join(root, filename))
        if not target.startswith(root + os.sep):
            raise PermissionError(f"Denied write outside {root}")
        return super().run(target, content)

class RestrictedFileReadTool(FileReadTool):
    base_dir: str = Field(..., description="Root dir for all reads")

    def run(self, filename: str):
        root = os.path.abspath(self.base_dir)
        target = os.path.abspath(os.path.join(root, filename))
        if not target.startswith(root + os.sep):
            raise PermissionError(f"Denied read outside {root}")
        return super().run(target)
    
