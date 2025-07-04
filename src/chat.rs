use anyhow::{Result, anyhow};
use ort::{Environment, Session, SessionBuilder, Value};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use ndarray::{ArrayD, CowArray};
use minijinja::context;
use std::fs;
use serde::{Deserialize, Serialize};
use serde_json;

#[derive(Debug, Deserialize, Serialize)]
struct ModelConfig {
    #[serde(default = "default_eos_token_id")]
    eos_token_id: u32,
    #[serde(default = "default_bos_token_id")]
    bos_token_id: u32,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

fn default_eos_token_id() -> u32 { 2 }
fn default_bos_token_id() -> u32 { 1 }
fn default_max_position_embeddings() -> usize { 2048 }
fn default_vocab_size() -> usize { 32000 }

pub struct ChatBot {
    session: Option<Session>,
    tokenizer: Tokenizer,
    model_loaded: bool,
    eos_token_id: u32,
    conversation_history: Vec<(String, String)>, // (user, assistant) pairs
    chat_template: String,
    config: ModelConfig,
}

impl ChatBot {
    pub async fn new() -> Result<Self> {
        println!("Initializing TinyLlama Chat Bot...");
        
        let environment = Arc::new(Environment::builder()
            .with_name("tinyllama_chat_bot")
            .build()?);

        let tokenizer_path = "models/tinyllama/tokenizer.json"; // Using native TinyLlama tokenizer
        let model_path = "models/tinyllama/model.onnx";
        let config_path = "models/tinyllama/config.json";
        
        // Load tokenizer
        println!("Loading native TinyLlama tokenizer (updated crate)...");
        let tokenizer = if Path::new(tokenizer_path).exists() {
            match Tokenizer::from_file(tokenizer_path) {
                Ok(tokenizer) => {
                    println!("✅ Native TinyLlama tokenizer loaded successfully!");
                    tokenizer
                },
                Err(e) => {
                    return Err(anyhow!("Failed to load tokenizer: {}", e));
                }
            }
        } else {
            return Err(anyhow!("Tokenizer not found at {}", tokenizer_path));
        };

        // Load model configuration
        println!("Loading model configuration...");
        let config_str = fs::read_to_string(config_path)
            .map_err(|e| anyhow!("Failed to load config.json: {}", e))?;
        let config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;
        println!("📋 Model config loaded: max_position_embeddings={}, eos_token_id={}", 
                 config.max_position_embeddings, config.eos_token_id);
        
        // Get EOS token ID from config (preferred) or tokenizer
        let eos_token_id = config.eos_token_id;
        println!("🔚 EOS token ID from config: {}", eos_token_id);
        
        // Load Jinja template
        let template_path = "models/tinyllama/chat_template.jinja";
        let chat_template = fs::read_to_string(template_path)
            .map_err(|e| anyhow!("Failed to load chat template: {}", e))?;

        // Try to load ONNX model
        println!("Looking for ONNX model at: {}", model_path);
        let (session, model_loaded) = if Path::new(model_path).exists() {
            println!("Loading TinyLlama ONNX model...");
            match SessionBuilder::new(&environment)?.with_model_from_file(model_path) {
                Ok(session) => {
                    println!("✅ TinyLlama ONNX model loaded successfully!");
                    println!("🎯 Full AI chat inference ready!");
                    (Some(session), true)
                },
                Err(e) => {
                    println!("⚠️ Failed to load ONNX model: {}", e);
                    return Err(anyhow!("Model loading failed: {}", e));
                }
            }
        } else {
            return Err(anyhow!("ONNX model not found at {}", model_path));
        };

        Ok(ChatBot {
            session,
            tokenizer,
            model_loaded,
            eos_token_id,
            conversation_history: Vec::new(),
            chat_template,
            config,
        })
    }

    pub async fn generate_response(&mut self, input: &str) -> Result<String> {
        // Special commands
        match input.to_lowercase().as_str() {
            s if s.contains("model") || s.contains("onnx") => {
                if self.model_loaded {
                    return Ok("✅ TinyLlama-1.1B-Chat ONNX model loaded! Using full AI chat inference.".to_string());
                } else {
                    return Ok("❌ TinyLlama model not loaded.".to_string());
                }
            },
            "clear" | "reset" => {
                self.conversation_history.clear();
                return Ok("🔄 Conversation history cleared!".to_string());
            },
            _ => {}
        }

        // Use real ONNX inference for chat
        if self.model_loaded {
            self.run_chat_inference(input).await
        } else {
            Err(anyhow!("Model not loaded"))
        }
    }

    async fn run_chat_inference(&mut self, input: &str) -> Result<String> {
        let session = self.session.as_ref().unwrap();
        
        // Build messages array for Jinja template
        let mut messages = vec![
            serde_json::json!({
                "role": "system",
                "content": "You are ChatBOT, a helpful, friendly, and knowledgeable AI assistant. Your name is ChatBOT. When introducing yourself, always use the name ChatBOT. Provide clear, detailed, and conversational responses. Be helpful and engaging while staying informative."
            })
        ];
        
        // Add conversation history for context (last 3 exchanges to avoid token limits)
        let recent_history = self.conversation_history
            .iter()
            .rev()
            .take(3)
            .rev();
            
        for (user_msg, assistant_msg) in recent_history {
            messages.push(serde_json::json!({
                "role": "user",
                "content": user_msg
            }));
            messages.push(serde_json::json!({
                "role": "assistant",
                "content": assistant_msg
            }));
        }
        
        // Add current user message
        messages.push(serde_json::json!({
            "role": "user",
            "content": input.trim()
        }));
        
        // Render the conversation using Jinja template
        let mut jinja_env = minijinja::Environment::new();
        jinja_env.add_template("chat", &self.chat_template)
            .map_err(|e| anyhow!("Failed to parse Jinja template: {}", e))?;
        
        let tmpl = jinja_env.get_template("chat")
            .map_err(|e| anyhow!("Failed to get template: {}", e))?;
        
        let conversation = tmpl.render(context! {
            messages => messages,
            eos_token => "</s>",
            add_generation_prompt => true
        }).map_err(|e| anyhow!("Failed to render template: {}", e))?;
        
        println!("🗨️ Building conversation with {} previous exchanges", self.conversation_history.len().min(3));
        println!("📝 Rendered template:\n{}", conversation);
        
        // Tokenize input
        let encoding = self.tokenizer.encode(conversation.as_str(), false)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        
        let mut current_tokens = encoding.get_ids().to_vec();
        println!("🔤 Tokenizing conversation → {} tokens", current_tokens.len());
        
        let mut generated_tokens = Vec::new();
        let max_new_tokens = 100; // Longer responses
        let temperature = 0.7; // Slightly more focused for better quality
        let top_p = 0.95; // Higher top_p for more diverse vocabulary
        
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Generate tokens one by one
        for step in 0..max_new_tokens {
            let seq_len = current_tokens.len();
            
            // Check token limit from config
            if seq_len > self.config.max_position_embeddings - 200 {
                println!("⚠️ Approaching token limit ({}/{}), stopping generation", 
                         seq_len, self.config.max_position_embeddings);
                break;
            }
            
            // Convert to ONNX tensors
            let input_ids_data: Vec<i64> = current_tokens.iter().map(|&x| x as i64).collect();
            let input_ids = ArrayD::from_shape_vec(vec![1, seq_len], input_ids_data)?;
            let input_ids_cow = CowArray::from(input_ids);

            let attention_mask_data: Vec<i64> = vec![1i64; seq_len];
            let attention_mask = ArrayD::from_shape_vec(vec![1, seq_len], attention_mask_data)?;
            let attention_mask_cow = CowArray::from(attention_mask);

            let position_ids_data: Vec<i64> = (0..seq_len as i64).collect();
            let position_ids = ArrayD::from_shape_vec(vec![1, seq_len], position_ids_data)?;
            let position_ids_cow = CowArray::from(position_ids);

            // Create ONNX inputs
            let input_tensor = Value::from_array(session.allocator(), &input_ids_cow)?;
            let attention_tensor = Value::from_array(session.allocator(), &attention_mask_cow)?;
            let position_tensor = Value::from_array(session.allocator(), &position_ids_cow)?;

            // Run inference
            if step == 0 {
                println!("🧠 Running TinyLlama chat inference...");
            }
            let outputs = session.run(vec![input_tensor, attention_tensor, position_tensor])?;
            
            // Extract logits for the last position
            let logits = outputs[0].try_extract::<f32>()?;
            let logits_view = logits.view();
            let next_pos = seq_len - 1;
            let prediction_logits = logits_view.slice(ndarray::s![0, next_pos, ..]);
            
            // Apply temperature scaling
            let scaled_logits: Vec<f32> = prediction_logits.iter()
                .map(|&logit| logit / temperature)
                .collect();
            
            // Apply softmax to get probabilities
            let max_logit = scaled_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut probs: Vec<f32> = scaled_logits.iter()
                .map(|&logit| (logit - max_logit).exp())
                .collect();
            let sum_probs: f32 = probs.iter().sum();
            for prob in &mut probs {
                *prob /= sum_probs;
            }
            
            // Create probability distribution for nucleus sampling
            let mut indexed_probs: Vec<(usize, f32)> = probs.iter()
                .enumerate()
                .map(|(i, &p)| (i, p))
                .collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            
            // Nucleus (top-p) sampling
            let mut cumulative_prob = 0.0;
            let mut nucleus_size = 0;
            for (_, prob) in &indexed_probs {
                cumulative_prob += prob;
                nucleus_size += 1;
                if cumulative_prob >= top_p {
                    break;
                }
            }
            
            // Sample from the nucleus
            let nucleus = &indexed_probs[..nucleus_size];
            let nucleus_sum: f32 = nucleus.iter().map(|(_, p)| p).sum();
            let random_value = rng.gen::<f32>() * nucleus_sum;
            
            let mut cumulative = 0.0;
            let mut selected_token = nucleus[0].0 as u32;
            for (token_id, prob) in nucleus {
                cumulative += prob;
                if cumulative >= random_value {
                    selected_token = *token_id as u32;
                    break;
                }
            }
            
            // Check for end-of-sequence token
            if selected_token == self.eos_token_id {
                println!("🔚 Hit EOS token, stopping generation");
                break;
            }
            
            // Add the new token
            generated_tokens.push(selected_token);
            current_tokens.push(selected_token);
            
            // Improved early stopping - look for natural conversation endings
            if generated_tokens.len() >= 5 {
                let partial_text = self.tokenizer.decode(&generated_tokens, true)
                    .unwrap_or_default();
                
                // Stop on natural endings, but require minimum length
                if generated_tokens.len() >= 10 && (
                    partial_text.trim().ends_with('.') || 
                    partial_text.trim().ends_with('!') || 
                    partial_text.trim().ends_with('?') ||
                    partial_text.contains("<|") // Stop if we hit chat markers
                ) {
                    break;
                }
                
                // Stop on repeated patterns (avoid loops)
                if generated_tokens.len() >= 20 {
                    let last_10: Vec<u32> = generated_tokens.iter().rev().take(10).cloned().collect();
                    let prev_10: Vec<u32> = generated_tokens.iter().rev().skip(10).take(10).cloned().collect();
                    if last_10 == prev_10 {
                        println!("🔁 Detected repetition, stopping");
                        break;
                    }
                }
            }
        }
        
        // Decode the generated tokens
        let generated_text = self.tokenizer.decode(&generated_tokens, true)
            .map_err(|e| anyhow!("Failed to decode: {}", e))?;
        
        let response = generated_text.trim();
        
        // Clean up any leftover chat markers or unwanted text
        let without_eos = response.replace("</s>", ""); // Remove EOS tokens
        let cleaned_response = without_eos
            .split("<|").next().unwrap_or(&without_eos) // Remove any chat markers
            .trim();
        
        println!("🎯 Generated {} tokens: '{}'", generated_tokens.len(), cleaned_response);
        
        let final_response = if cleaned_response.is_empty() { 
            "I'm here to help! What would you like to know?".to_string()
        } else { 
            cleaned_response.to_string()
        };
        
        // Store this exchange in conversation history
        self.conversation_history.push((input.to_string(), final_response.clone()));
        
        // Keep only last 5 exchanges to manage memory
        if self.conversation_history.len() > 5 {
            self.conversation_history.drain(0..self.conversation_history.len() - 5);
        }
        
        Ok(final_response)
    }
}
