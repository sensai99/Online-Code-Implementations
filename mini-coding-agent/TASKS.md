# Goal: Given a user query accomplish the task.

## Questions:

- What kind of queries can the system solve?:
    
    - Basic creation/edit/deletion of a file - DONE

- Does it support multi-turn conversations? - Start with single-turn conversations (i.e. there is no context from history stored)

- What kind of tools does it need?

    - File edit operation (editing) - Given a single file path, edit the content

- How is the system restricted from using certain tools/performing certain actions? - Always ask for approval for all the actions that it plans to take

    - Ask for approval when deleting file - DONE

- How is the system evaluated?

    - Final user feedback

    - Run some unit tests on whether the final code runs properly or not (not just the code that it changed but the other code in the file - if it edits the file)

    - How much time did it take? i.e. the number of operations that it invoked


## Features:

- The agentic system should be more interactive and not independent - i.e. it should first show what it's planning to do and then ask approval from the user whether it should proceed

