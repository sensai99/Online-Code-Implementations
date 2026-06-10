def get_approval():
    response = input(f"Are you sure you want to proceed with the changes? (y/n): ")
    if response == "y":
        return True
    else:
        return False