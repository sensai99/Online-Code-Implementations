def get_approval(action=None, kwargs=None):
    if action == "create_file":
        response = input(f"Are you sure you want to create the file {kwargs['file_path']}? (y/n): ")
    elif action == "edit_file":
        response = input(f"Are you sure you want to edit the file {kwargs['file_path']}? (y/n): ")
    elif action == "delete_file":
        response = input(f"Are you sure you want to delete the file {kwargs['file_path']}? (y/n): ")
    else:
        response = input(f"Are you sure you want to proceed with the changes? (y/n): ")
    
    if response == "y":
        return True
    else:
        return False