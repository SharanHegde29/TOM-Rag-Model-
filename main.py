import sys
from tom_agent import Tom

def main():
    print("Booting up TOM Core Architecture...")
    tom = Tom()
    
    # Check if it loaded past memories (subtracting 1 because the system prompt is always there)
    memory_count = len(tom.memory) - 1
    
    print("========================================")
    print("      TOM — ONLINE         ")
    print(f"      [Memories Restored: {memory_count}]")
    print("========================================\n")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nTOM: Archiving memories... Powering down. Catch you later, Sharan!")
            
            # THE MAGIC SAVE COMMAND
            tom.save_memory() 
            break
            
        print("\nTOM: ", end="")
        stream = tom.chat(user_input)
        
        full_reply = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                sys.stdout.write(token)
                sys.stdout.flush()
                full_reply += token
                
        tom.memory.append({"role": "assistant", "content": full_reply})
        print() 

if __name__ == "__main__":
    main()