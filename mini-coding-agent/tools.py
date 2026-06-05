class EditFileTool:
    def __init__(self):
        self.name = "edit_file"
        self.description = "Edit a file"
    
    def edit_file(self, file_path, content):
        with open(file_path, "a") as f:
            f.write(content)
        return "File edited successfully"

    def __call__(self, file_path, content):
        return self.edit_file(file_path, content)

class CreateFileTool:
    def __init__(self):
        self.name = "create_file"
        self.description = "Create a new file"
    
    def create_file(self, file_path, content):
        with open(file_path, "w") as f:
            f.write(content)
        return "File created successfully"

    def __call__(self, file_path, content):
        return self.create_file(file_path, content)

class NoopTool:
    def __init__(self):
        self.name = "noop"
        self.description = "Do not use any tools"
    
    def __call__(self, **kwargs):
        return "Sorry, looks like your query was incomplete to accomplish the task. Please try again with a more specific query."

class ToolsManager:
    def __init__(self):
        self.tools = {
            "edit_file": EditFileTool(),
            "create_file": CreateFileTool(),
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