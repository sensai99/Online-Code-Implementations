import run as run_agent
from utils.file_utils import save_jsonl, load_jsonl

def main():
    print("Welcome to the mini coding agent!")
    print("Ask me anything about coding!")

    context = load_jsonl("context/history.jsonl")

    while True:
        user_input = input("You: ")
        context.append({"role": "user", "content": user_input})
        
        # exit if user wants to exit
        if user_input.lower() == "exit":
            break
            
        print("Agent: \n")
        new_context = run_agent.run(context)
        save_jsonl([context[-1], *new_context], "context/history.jsonl")
        context.extend(new_context)

if __name__ == '__main__':
    main()