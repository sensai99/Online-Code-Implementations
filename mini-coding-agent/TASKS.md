Goal: Given a user query accomplish the task.

Questions:

- What kind of queries can the system solve? - Writing/Edit/Modify a code snippet (from a file or create a new file)

- Does it support multi-turn conversations? - Start with single-turn conversations (i.e. there is no context from history stored)

- What kind of tools does it need?

    - File edit operation (editing) - Given a single file path, edit the content

- How is the system restricted from using certain tools/performing certain actions? - Always ask for approval for all the actions that it plans to take

- How is the system evaluated?

    - Final user feedback

    - Run some unit tests on whether the final code runs properly or not (not just the code that it changed but the other code in the file - if it edits the file)

    - How much time did it take? i.e. the number of operations that it invoked