use std::io::{self, Write, IsTerminal, Read};
use anyhow::Result;
use std::env;

mod chat;

#[tokio::main]
async fn main() -> Result<()> {
    // Get model name from command line args or default to tinyllama
    let args: Vec<String> = env::args().collect();
    let model_name = args.get(1).map(|s| s.as_str()).unwrap_or("tinyllama");
    
    let mut chat_bot = chat::ChatBot::new(model_name).await?;
    
    // Check if input is piped or interactive
    if io::stdin().is_terminal() {
        // Interactive mode
        println!("🤖 ONNX Chat Bot - Type 'quit' to exit");
        println!("Loading model...");
        
        loop {
            print!("\n> ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            
            if input.is_empty() {
                continue;
            }
            
            if input.to_lowercase() == "quit" {
                println!("Goodbye!");
                break;
            }
            
            let response = chat_bot.generate_response(input).await?;
            println!("🤖: {}", response);
        }
    } else {
        // Piped input mode
        let mut input = String::new();
        io::stdin().read_to_string(&mut input)?;
        let input = input.trim();
        
        if !input.is_empty() {
            let response = chat_bot.generate_response(input).await?;
            println!("{}", response);
        }
    }
    
    Ok(())
}
