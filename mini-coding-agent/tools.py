import os
from utils.tool_utils import get_approval

class EditFileTool:
    def __init__(self):
        self.name = "edit_file"
        self.description = "Edit a file"
    
    def edit_file(self, file_path, content):
        if get_approval(action="edit_file", kwargs={"file_path": file_path}):
            with open(file_path, "a") as f:
                f.write(content)
            return "File edited successfully, as user approved the changes"
        else:
            return "File not edited, as user did not approve the changes"

    def __call__(self, file_path, content):
        return self.edit_file(file_path, content)

class CreateFileTool:
    def __init__(self):
        self.name = "create_file"
        self.description = "Create a new file"
    
    def create_file(self, file_path, content):
        if get_approval(action="create_file", kwargs={"file_path": file_path}):
            with open(file_path, "w") as f:
                f.write(content)
            return "File created successfully, as user approved the changes"
        else:
            return "File not created, as user did not approve the changes"

    def __call__(self, file_path, content):
        return self.create_file(file_path, content)

class DeleteFileTool:
    def __init__(self):
        self.name = "delete_file"
        self.description = "Delete a file"
        self.confirm = False
    
    def delete_file(self, file_path):
        if os.path.exists(file_path):
            if get_approval(action="delete_file", kwargs={"file_path": file_path}):
                os.remove(file_path)
                return "File deleted successfully, as user approved the changes"
            else:
                return "File not deleted, as user did not approve the changes"
        else:
            return "File does not exist!"
    
    def __call__(self, file_path):
        return self.delete_file(file_path)

class NoopTool:
    def __init__(self):
        self.name = "noop"
        self.description = "Do not use any tools"
    
    def __call__(self, **kwargs):
        return ""

class ToolsManager:
    def __init__(self):
        self.tools = {
            "edit_file": EditFileTool(),
            "create_file": CreateFileTool(),
            "delete_file": DeleteFileTool(),
            "noop": NoopTool()
        }

    def add_tool(self, tool):
        self.tools.append(tool)

    def get_tools(self):
        return self.tools

    def get_tool(self, name):
        return self.tools[name]
    
    def __call__(self, name, **kwargs):
        return self.tools[name](**kwargs)