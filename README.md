# 🤖 AI Live Chat Agent for Online Stores

This is an open-source AI-powered customer service agent designed to handle live chat for online shops. The agent can answer customer questions based on a knowledge base and simulate order lookup and processing. It is lightweight, easy to set up, and customizable for developers and small businesses.

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- OpenAI API key

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ai-live-chat.git
   cd ai-live-chat
   ```

2. **Set up virtual environment**

   **For macOS/Linux:**
   ```bash
   # Make the setup script executable
   chmod +x setup.sh
   
   # Run the setup script
   ./setup.sh
   ```

   **For Windows:**
   ```bash
   setup.bat
   ```

3. **Configure environment variables**
   ```bash
   # Copy the example environment file
   cp .env.example .env
   
   # Edit .env and add your OpenAI API key
   # OPENAI_API_KEY=your-api-key-here
   ```

4. **Run the test script**
   ```bash
   # Make sure your virtual environment is activated
   source venv/bin/activate  # For macOS/Linux
   # or
   venv\Scripts\activate.bat  # For Windows
   
   # Run the test script
   python test_agent.py
   ```

### Troubleshooting

#### SSL Certificate Issues
If you encounter SSL certificate errors during package installation, you can try the following:

1. **Manual Installation**
   ```bash
   # Activate virtual environment first
   source venv/bin/activate  # For macOS/Linux
   # or
   venv\Scripts\activate.bat  # For Windows
   
   # Install packages manually with trusted hosts
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt
   ```

2. **Update Certificates**
   ```bash
   # For macOS
   /Applications/Python\ 3.10/Install\ Certificates.command
   
   # For Linux
   sudo apt-get update
   sudo apt-get install ca-certificates
   ```

3. **Alternative Installation Method**
   ```bash
   # Install packages one by one
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org langchain>=0.1.4
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org openai>=1.6.1
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org python-dotenv>=1.0.0
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org faiss-cpu>=1.7.4
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org sentence-transformers>=2.2.2
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org tiktoken>=0.5.2
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org pydantic>=2.5.3
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org typing-extensions>=4.9.0
   pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy>=1.26.2
   ```

## 🚀 Usage

### Running the Test Agent

The test agent can be run with different configurations using command-line arguments:

```bash
# Basic usage with OpenAI
python test_agent.py

# Using OpenRouter with specific model
python test_agent.py --llm openrouter --model "openai/gpt-4o-mini"

# Using HuggingFace embeddings
python test_agent.py --embeddings huggingface

# Custom temperature and model
python test_agent.py --llm openai --model "gpt-4o-mini" --temperature 0.8

# Full configuration example
python test_agent.py \
    --llm openrouter \
    --embeddings huggingface \
    --model "openai/gpt-4o-mini" \
    --temperature 0.7
```

### Command Line Arguments

- `--llm`: LLM provider to use (choices: 'openai', 'openrouter', default: 'openai')
- `--embeddings`: Embedding provider to use (choices: 'openai', 'huggingface', default: 'openai')
- `--model`: Specific model name to use (default: based on provider)
- `--temperature`: Temperature for model responses (default: 0.7)

### Example Interactions

```bash
$ python test_agent.py --llm openai
AI Chat Agent initialized with OpenAI
Type 'exit' to end the conversation

You: What are your shipping policies?
AI: Based on our shipping policies, we offer standard shipping (3-5 business days) and express shipping (1-2 business days). All orders over $50 qualify for free standard shipping. International shipping is available to select countries with delivery times varying by location.

You: How do I return an item?
AI: To return an item, you need to initiate the return within 30 days of delivery. Please contact our customer service with your order number and reason for return. Once approved, you'll receive a return shipping label. Items must be in original condition with all tags attached.

You: exit
Goodbye!
```

## 🎯 Project Goals

Many online stores lack 24/7 customer support or cannot handle a high volume of inquiries. This project aims to:

- Provide an easily deployable AI chat agent for e-commerce websites
- Answer common customer questions using a knowledge base
- Simulate basic order inquiry and handling
- Allow developers to extend or integrate the agent into their own platforms

## 🧠 Design Overview

The agent is designed using a **Retrieval-Augmented Generation (RAG)** architecture:

1. **Knowledge Retrieval**

   - When a user asks a question, the system retrieves relevant information from the knowledge base
   - A language model generates a natural-sounding response based on the retrieved context

2. **Flexible Knowledge Base Format**

   - Supports Markdown (`.md`), plain text (`.txt`), or JSON documents
   - Store-specific content such as FAQs, return policies, and shipping terms can be easily added

3. **Simulated Order Lookup**

   - Customers can ask questions about orders using their email or order number
   - The system fetches mock order data and responds with appropriate information

4. **Conversational Tone**
   - Designed to respond in English with a friendly, professional tone
   - Cantonese is also supported for stores targeting Hong Kong or Cantonese-speaking customers

## 🛠️ Technical Stack

| Component             | Technology                                                                                          |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| Programming Language  | Python 3.10+                                                                                        |
| AI Framework          | [LangChain](https://www.langchain.com/), [OpenAI GPT](https://platform.openai.com/) (or other LLMs) |
| Vector DB / Retrieval | [FAISS](https://github.com/facebookresearch/faiss) or Chroma DB (pluggable)                         |
| Knowledge Base Format | Markdown, TXT, or JSON                                                                              |
| Frontend              | React with TypeScript                                                                               |
| Backend API           | FastAPI                                                                                            |
| Database              | PostgreSQL, Redis                                                                                   |
| Deployment            | Docker, Cloud Platforms (AWS/GCP)                                                                   |

## 🔌 Integration Design

The AI chat agent is designed to be easily integrated into any existing website using an iframe-based approach. This design provides maximum compatibility and isolation while maintaining a seamless user experience.

### Architecture Overview

```
[Company's Website]                    [AI Chat Project]
+------------------+                  +------------------+
|                  |                  |                  |
|  Company's       |                  |  Backend API     |
|  Existing        |                  |  Server          |
|  Website         |                  |  (Hosted)        |
|                  |                  |                  |
|  +------------+  |                  |  +------------+  |
|  |            |  |                  |  |            |  |
|  |  iframe    |<------------------->|  |  Chat      |  |
|  |  embed     |  |                  |  |  Frontend  |  |
|  |            |  |                  |  |  (Hosted)  |  |
|  +------------+  |                  |  +------------+  |
|                  |                  |                  |
+------------------+                  +------------------+
```

### Integration Methods

#### 1. Basic Iframe Integration
```html
<iframe
  src="https://chat.your-domain.com/embed?apiKey=your-api-key"
  style="width: 350px; height: 500px; border: none;"
></iframe>
```

#### 2. Responsive Container
```html
<div class="chat-container" style="position: fixed; bottom: 20px; right: 20px; width: 350px; height: 500px; z-index: 1000;">
  <iframe
    src="https://chat.your-domain.com/embed?apiKey=your-api-key"
    style="width: 100%; height: 100%; border: none; border-radius: 10px; box-shadow: 0 2px 12px rgba(0,0,0,0.15);"
  ></iframe>
</div>
```

### Configuration Options

The chat widget can be customized through URL parameters:

```javascript
const config = {
  // Basic settings
  apiKey: 'your-api-key',
  theme: 'light',
  language: 'en',
  
  // Customization
  customStyles: {
    primaryColor: '#007bff',
    backgroundColor: '#ffffff',
    fontFamily: 'Arial, sans-serif'
  },
  
  // Features
  features: {
    fileUpload: true,
    voiceInput: true,
    emojiPicker: true
  }
};

// Apply configuration
const chatUrl = new URL('https://chat.your-domain.com/embed');
Object.entries(config).forEach(([key, value]) => {
  chatUrl.searchParams.append(key, JSON.stringify(value));
});
```

### Deployment Options

1. **Self-Hosted**
   - Host the backend and frontend on your own servers
   - Maintain your own API keys
   - Full control over the infrastructure

2. **Cloud-Hosted**
   - Use our hosted solution
   - Get your own subdomain
   - Automatic updates and maintenance

3. **Hybrid**
   - Host the frontend on our servers
   - Host the backend on your servers
   - More control while maintaining easy updates

### Security Features

- API key authentication
- Content Security Policy (CSP) support
- Origin validation
- Secure communication between parent and iframe
- Isolated chat environment

### Benefits

1. **Easy Integration**
   - Single iframe tag to add
   - No need to modify existing code
   - Works with any website platform

2. **Isolation**
   - Chat functionality is isolated
   - No interference with existing website
   - Secure communication

3. **Customization**
   - Customizable appearance
   - Configurable features
   - Brand consistency

4. **Maintenance**
   - Automatic updates
   - Centralized management
   - Easy troubleshooting

## 📚 Knowledge Base Example

You can create your knowledge base using Markdown (`.md`), plain text (`.txt`), or JSON files. Each file can contain descriptive text, FAQs, or structured data (e.g. tables). The AI agent will parse and retrieve relevant information to answer customer questions.

Place your content in the `/knowledge_base/` directory. Here are some examples:

---

### 📦 `shipping_info.md`

```markdown
# Shipping Information

We offer delivery to multiple regions. Below is the estimated delivery time and cost:

| Region        | Delivery Time      | Standard Shipping | Express Shipping |
| ------------- | ------------------ | ----------------- | ---------------- |
| Hong Kong     | 1–2 business days  | Free              | $30 HKD          |
| Macau         | 2–3 business days  | $20 HKD           | $50 HKD          |
| Taiwan        | 3–5 business days  | $30 HKD           | $80 HKD          |
| International | 5–10 business days | $60 HKD           | $150 HKD         |

> Note: Shipping times may vary during public holidays or extreme weather conditions.
```
### 🤝  `return_policy.md`

```markdown
# Return & Refund Policy

We want you to be happy with your purchase.

- You may request a return within **7 days** of receiving your order.
- Items must be unused and in original packaging.
- To request a return, please contact our support team at `support@example.com`.

Refunds will be processed within 3–5 business days after we receive the returned item.
```


### ❓ `faq.md`

```markdown
# Frequently Asked Questions

## Q: What payment methods do you accept?
A: We accept Visa, MasterCard, PayPal, and Apple Pay.

## Q: How can I track my order?
A: Once your order is shipped, you will receive an email with a tracking link.

## Q: Can I change the shipping address after ordering?
A: Yes, but only if the order hasn't been shipped yet. Please contact our support team as soon as possible.
```

## 📁 Project Structure

```bash
.
│
├── knowledge_base/         # Custom knowledge base files
│   ├── faq.md
│   ├── return_policy.md
│   └── shipping_info.md
│
├── data/orders.json        # Mock order records for testing
│
└── README.md               # Project documentation
```

## 🔧 Configuration

The agent can be configured using environment variables. Copy the `.env.example` file to `.env` and modify the values as needed:

```bash
cp .env.example .env
```

### Environment Variables

#### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key (if using OpenRouter)
- `HUGGINGFACE_API_KEY`: Your HuggingFace API key (if using HuggingFace embeddings)

#### Model Configuration
- `OPENAI_MODEL_NAME`: OpenAI model to use (default: "gpt-4o-mini")
- `OPENROUTER_MODEL_NAME`: OpenRouter model to use (default: "openai/gpt-4o-mini")
- `TEMPERATURE`: Model temperature (default: 0.7)
- `MAX_TOKENS`: Maximum tokens per response (default: 2000)

#### Knowledge Base Configuration
- `KNOWLEDGE_BASE_DIR`: Directory containing knowledge base files (default: "./knowledge_base")
- `VECTOR_STORE_PATH`: Where to store the FAISS index (optional)
- `CHUNK_SIZE`: Size of text chunks for processing (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)

#### Logging Configuration
- `LOG_LEVEL`: Logging level (default: "INFO")
- `VERBOSE`: Enable verbose output (default: "true")

### Example Configuration

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL_NAME=gpt-4o-mini

# OpenRouter Configuration
OPENROUTER_API_KEY=sk-...
OPENROUTER_MODEL_NAME=openai/gpt-4o-mini

# Agent Configuration
TEMPERATURE=0.7
MAX_TOKENS=2000

# Knowledge Base Configuration
KNOWLEDGE_BASE_DIR=./knowledge_base
VECTOR_STORE_PATH=./vector_store
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Logging Configuration
LOG_LEVEL=INFO
VERBOSE=true
```

