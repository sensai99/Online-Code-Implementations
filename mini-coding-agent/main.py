import run as run_agent

def main():
    print("Welcome to the mini coding agent!")
    print("Ask me anything about coding!")

    while True:
        user_input = input("You: ")

        # exit if user wants to exit
        if user_input.lower() == "exit":
            break
            
        print("Agent: \n")
        response = run_agent.run(user_input)
        print(response)

if __name__ == '__main__':
    main()