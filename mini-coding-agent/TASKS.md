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


## TODOs:

- The agentic system should be more interactive and not independent - i.e. it should first show what it's planning to do and then ask approval from the user whether it should proceed

    - It is now more interactive - responds it's thoughts about the user message instead of just performing the action.

- Context Management: Use the entire conversation history as context

- Add read_file tool, so that system can inspect existing files to fix bugs, etc.

- format for context in prompts.py doesn't match with what we are passing (JSON Object instead of the specified format). This is broken, fix it

- any failures (e.g. file parsing, etc) -> have a retry mechanism

- run a tool call for verification of code
