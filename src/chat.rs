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
    #[serde(default = "default_max_position_embeddings")]
    n_positions: usize,  // GPT2 style
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
}

#[derive(Debug, Deserialize, Serialize)]
struct TokenizerConfig {
    #[serde(default)]
    add_bos_token: bool,
    #[serde(default)]
    add_eos_token: bool,
    #[serde(default)]
    bos_token: Option<String>,
    #[serde(default)]
    eos_token: Option<String>,
    #[serde(default = "default_model_max_length")]
    model_max_length: usize,
    #[serde(default)]
    pad_token: Option<String>,
}

fn default_model_max_length() -> usize { 1024 }

fn default_eos_token_id() -> u32 { 2 }
fn default_bos_token_id() -> u32 { 1 }
fn default_max_position_embeddings() -> usize { 2048 }
fn default_vocab_size() -> usize { 32000 }

#[derive(Debug, Clone)]
struct SpecialToken {
    content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
}

impl SpecialToken {
    fn from_value(value: &serde_json::Value) -> Option<Self> {
        match value {
            serde_json::Value::String(s) => Some(SpecialToken {
                content: s.clone(),
                lstrip: false,
                normalized: false,
                rstrip: false,
                single_word: false,
            }),
            serde_json::Value::Object(obj) => Some(SpecialToken {
                content: obj.get("content")?.as_str()?.to_string(),
                lstrip: obj.get("lstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                normalized: obj.get("normalized").and_then(|v| v.as_bool()).unwrap_or(false),
                rstrip: obj.get("rstrip").and_then(|v| v.as_bool()).unwrap_or(false),
                single_word: obj.get("single_word").and_then(|v| v.as_bool()).unwrap_or(false),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct SpecialTokensMap {
    #[serde(default)]
    bos_token: Option<serde_json::Value>,
    #[serde(default)]
    eos_token: Option<serde_json::Value>,
    #[serde(default)]
    unk_token: Option<serde_json::Value>,
    #[serde(default)]
    pad_token: Option<serde_json::Value>,
}

pub struct ChatBot {
    session: Option<Session>,
    tokenizer: Tokenizer,
    model_loaded: bool,
    eos_token_id: u32,
    bos_token_id: u32,
    conversation_history: Vec<(String, String)>, // (user, assistant) pairs
    chat_template: String,
    config: ModelConfig,
    tokenizer_config: TokenizerConfig,
    special_tokens_map: SpecialTokensMap,
    eos_token_str: String,
    bos_token_str: String,
    model_name: String,
}

impl ChatBot {
    pub async fn new(model_name: &str) -> Result<Self> {
        println!("Initializing {} Chat Bot...", model_name);
        
        let environment = Arc::new(Environment::builder()
            .with_name("onnx_chat_bot")
            .build()?);

        let model_dir = format!("models/{}", model_name);
        let tokenizer_path = format!("{}/tokenizer.json", model_dir);
        let model_path = format!("{}/model.onnx", model_dir);
        let config_path = format!("{}/config.json", model_dir);
        let tokenizer_config_path = format!("{}/tokenizer_config.json", model_dir);
        let special_tokens_map_path = format!("{}/special_tokens_map.json", model_dir);
        let template_path = format!("{}/chat_template.jinja", model_dir);
        
        // Load tokenizer
        println!("Loading tokenizer for {}...", model_name);
        let tokenizer = if Path::new(&tokenizer_path).exists() {
            match Tokenizer::from_file(&tokenizer_path) {
                Ok(tokenizer) => {
                    println!("✅ Tokenizer loaded successfully!");
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
        let mut config: ModelConfig = serde_json::from_str(&config_str)
            .map_err(|e| anyhow!("Failed to parse config.json: {}", e))?;
        
        // Handle different naming conventions
        if config.max_position_embeddings == 2048 && config.n_positions != 2048 {
            config.max_position_embeddings = config.n_positions;
        }
        
        println!("📋 Model config loaded: max_position_embeddings={}, eos_token_id={}", 
                 config.max_position_embeddings, config.eos_token_id);
        
        // Load tokenizer config
        println!("Loading tokenizer configuration...");
        let tokenizer_config_str = fs::read_to_string(tokenizer_config_path)
            .map_err(|e| anyhow!("Failed to load tokenizer_config.json: {}", e))?;
        let tokenizer_config: TokenizerConfig = serde_json::from_str(&tokenizer_config_str)
            .map_err(|e| anyhow!("Failed to parse tokenizer_config.json: {}", e))?;
        println!("📖 Tokenizer config loaded: add_bos_token={}, model_max_length={}", 
                 tokenizer_config.add_bos_token, tokenizer_config.model_max_length);
        
        // Load special tokens map
        println!("Loading special tokens map...");
        let special_tokens_map_str = fs::read_to_string(special_tokens_map_path)
            .map_err(|e| anyhow!("Failed to load special_tokens_map.json: {}", e))?;
        let special_tokens_map: SpecialTokensMap = serde_json::from_str(&special_tokens_map_str)
            .map_err(|e| anyhow!("Failed to parse special_tokens_map.json: {}", e))?;
        
        // Get actual token strings from special_tokens_map
        let eos_token_str = special_tokens_map.eos_token.as_ref()
            .and_then(|v| SpecialToken::from_value(v))
            .map(|t| t.content)
            .unwrap_or_else(|| "</s>".to_string());
            
        let bos_token_str = special_tokens_map.bos_token.as_ref()
            .and_then(|v| SpecialToken::from_value(v))
            .map(|t| t.content)
            .unwrap_or_else(|| "<s>".to_string());
            
        println!("📑 Special tokens loaded: BOS='{}', EOS='{}'", bos_token_str, eos_token_str);
        
        // Get token IDs from config
        let eos_token_id = config.eos_token_id;
        let bos_token_id = config.bos_token_id;
        println!("🔚 Token IDs: EOS={}, BOS={}", eos_token_id, bos_token_id);
        
        // Load Jinja template (optional)
        let chat_template = if Path::new(&template_path).exists() {
            println!("📝 Loading chat template...");
            fs::read_to_string(&template_path)
                .map_err(|e| anyhow!("Failed to load chat template: {}", e))?
        } else {
            println!("📝 No chat template found, using default");
            // Default template for models without chat_template.jinja
            String::from("{% for message in messages %}{{ message['content'] }}{% if not loop.last %} {% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}")
        };

        // Try to load ONNX model
        println!("Looking for ONNX model at: {}", model_path);
        let (session, model_loaded) = if Path::new(&model_path).exists() {
            println!("Loading {} ONNX model...", model_name);
            match SessionBuilder::new(&environment)?.with_model_from_file(&model_path) {
                Ok(session) => {
                    println!("✅ {} ONNX model loaded successfully!", model_name);
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
            bos_token_id,
            conversation_history: Vec::new(),
            chat_template,
            config,
            tokenizer_config,
            special_tokens_map,
            eos_token_str,
            bos_token_str,
            model_name: model_name.to_string(),
        })
    }

    pub async fn generate_response(&mut self, input: &str) -> Result<String> {
        // Special commands
        match input.to_lowercase().as_str() {
            s if s.contains("model") || s.contains("onnx") => {
                if self.model_loaded {
                    return Ok(format!("✅ {} ONNX model loaded! Using full AI chat inference.", self.model_name));
                } else {
                    return Ok(format!("❌ {} model not loaded.", self.model_name));
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
        let mut messages = vec![];
        
        // Check if this is a chat model.
        // The heuristic is: if the template is not the hardcoded default, then it's a chat model.
        let is_chat_model = self.chat_template != String::from("{% for message in messages %}{{ message['content'] }}{% if not loop.last %} {% endif %}{% endfor %}{% if add_generation_prompt %}{% endif %}");
        
        // Only add system message for chat models that support the 'system' role
        if is_chat_model && self.chat_template.contains("system") {
            messages.push(serde_json::json!({
                "role": "system",
                "content": "You are ChatBOT, a helpful, friendly, and knowledgeable AI assistant. Your name is ChatBOT. When introducing yourself, always use the name ChatBOT. Provide clear, detailed, and conversational responses. Be helpful and engaging while staying informative."
            }));
        }
        
        // Add conversation history for context (last 3 exchanges to avoid token limits)
        if is_chat_model {
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
        } else {
            // For non-chat models, just use the input as a single message
            messages.push(serde_json::json!({
                "role": "user",
                "content": input.trim()
            }));
        }
        
        // Render the conversation using Jinja template
        let mut jinja_env = minijinja::Environment::new();
        jinja_env.add_template("chat", &self.chat_template)
            .map_err(|e| anyhow!("Failed to parse Jinja template: {}", e))?;
        
        let tmpl = jinja_env.get_template("chat")
            .map_err(|e| anyhow!("Failed to get template: {}", e))?;
        
        let conversation = tmpl.render(context! {
            messages => messages,
            eos_token => &self.eos_token_str,
            add_generation_prompt => true
        }).map_err(|e| anyhow!("Failed to render template: {}", e))?;
        
        println!("🗨️ Building conversation with {} previous exchanges", self.conversation_history.len().min(3));
        println!("📝 Rendered template:\n{}", conversation);
        
        // Tokenize input with proper BOS handling
        let add_special_tokens = self.tokenizer_config.add_bos_token;
        let encoding = self.tokenizer.encode(conversation.as_str(), add_special_tokens)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;
        
        let mut current_tokens = encoding.get_ids().to_vec();
        
        // If tokenizer didn't add BOS but config says we should, add it manually
        if self.tokenizer_config.add_bos_token && !current_tokens.is_empty() && current_tokens[0] != self.bos_token_id {
            current_tokens.insert(0, self.bos_token_id);
        }
        
        println!("🔤 Tokenizing conversation → {} tokens (BOS: {})", 
                 current_tokens.len(), 
                 if self.tokenizer_config.add_bos_token { "added" } else { "not added" });
        
        let mut generated_tokens = Vec::new();
        let max_new_tokens = 100; // Longer responses
        let temperature = 0.1; // Much lower temperature for deterministic output
        let top_p = 0.1; // Very focused sampling for consistency
        
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
                println!("🧠 Running {} chat inference...", self.model_name);
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
