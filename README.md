# Generative AI Architectures with LLM, Prompt, RAG, Vector DB

- ![alt text](image.png)
- ![alt text](image-1.png)
- ![alt text](image-2.png)
- ![alt text](image-3.png)
- ![alt text](image-4.png)
- ![alt text](image-5.png)
- ![alt text](image-6.png)
- ![alt text](image-7.png)
- ![alt text](image-8.png)
- ![alt text](image-9.png)

### Overview of EShop Customer Support System
- ![alt text](image-10.png)
- ![alt text](image-11.png)

## Generative AI 
- ![alt text](image-12.png)
- ![alt text](image-13.png)
- ![alt text](image-14.png)
- ![alt text](image-15.png)
- ![alt text](image-16.png)
- ![alt text](image-17.png)

### What is Generative AI
- ![alt text](image-19.png)
- ![alt text](image-20.png)
- ![alt text](image-21.png)
- For example, in marketing, generative AI can write product descriptions or design unique advertisements at scale.
- In music, it can compose new songs, and in software development it can even help to generate code.
- Generative AI is a breakthrough technology that enables machines to create new content based on what they have learned.

### How Generative AI works
- ![alt text](image-22.png)
- ![alt text](image-23.png)
- ![alt text](image-24.png)
- ![alt text](image-25.png)
- ![alt text](image-26.png)
- ![alt text](image-27.png)

### Types of Generative AI Models
- ![alt text](image-28.png)
- ![alt text](image-29.png)
- ![alt text](image-30.png)
- ![alt text](image-31.png)
- ![alt text](image-32.png)

### Transformer Architecture
- ![alt text](image-33.png)
- ![alt text](image-34.png)
- ![alt text](image-35.png)
- ![alt text](image-36.png)
- For a sentence like "The cat sat on the mat":
- When processing "cat," self-attention figures out how much focus to put on "The," "sat," "on," "the," and "mat" to understand "cat" better.
- It assigns attention scores to each word based on their relevance.
- Self-attention uses three vectors for each input token: Query (Q), Key (K), and Value (V). These are derived from the input embeddings via learned linear transformations.
- Self-attention is like a smart voting system: each word asks, "Who’s important to me?" (Query), checks "Who matches?" (Key), and builds a new meaning (Value) based on the answers—all in one go.
- ![alt text](image-37.png)

## What are Large Language Models(LLMs)
- ![alt text](image-38.png)
- ![alt text](image-39.png)
- ![alt text](image-40.png)

### How LLMs work?
- ![alt text](image-41.png)
- ![alt text](image-42.png)
- ![alt text](image-43.png)
- ![alt text](image-44.png)
- ![alt text](image-45.png)
- ![alt text](image-46.png)

### What is Token?
- ![alt text](image-48.png)
#### What is Tokenization ?
- ![alt text](image-49.png)
- ![alt text](image-50.png)
- ![alt text](image-51.png)

#### How LLMs use Tokens?
- ![alt text](image-52.png)
- ![alt text](image-53.png)
- ![alt text](image-54.png)

#### Capabilities of LLMs
- ![alt text](image-55.png)
- ![alt text](image-56.png)
- LLMs can analyze social media, post product reviews, or survey responses in order to determine overall customer sentiment.
- Instead of searching for exact phrases, semantic search can find documents or articles related to a query based on meaning, even if the words don't exactly match.
- This is widely used in document retrieval systems and recommendation engines.

#### LLM Use Cases and Real-World Applications
- ![alt text](image-57.png)
- ![alt text](image-58.png)

#### Limitations of LLMs
- ![alt text](image-59.png)
- ![alt text](image-60.png)

#### LLM Settings
- ![alt text](image-61.png)
#### Temperature
- The temperature setting controls how creative or random the model's response will be. It determines the degree of uncertainty in the model's choice.
- So if you select the lower level of the temperature, that makes the model response more focused, predictable and deterministic.
- And this is ideal for tasks where you want to accuracy like math problems or factual answers.
- But when you select a higher temperature, that makes the model more creative and diverse in its response.
- And this is useful for generating creative content like story writing or poetry.
#### Max Tokens: 
- This will be also called max length into the R, OpenAI playground and max tokens defines the maximum number of tokens that the model can generate.
- This is important because it directly impacts how long the response will be.
- For example, if you make this maximum length, a higher level, higher max token allows the model produce longer responses, which is useful for tasks like generating article or detailed summaries.
- And if you go back to lower values and restrict the response to shorter outputs, which can be helpful when you want to concise and summarize answers.
#### Stop Sequences
- We use stop sequences for specific phrases or characters that tell the model to stop generating text once they are encountered, so this is useful for situations where you want to control the length or structure of the output.
- For example, if you are creating a dialogue, you might set the stop sequence as a user so the model stops after reach the response and does not continue the conversation whenever see the user.

#### Top P:
- Top P, top P, or nucleus sampling control the diversity of the model's output by limiting the token choices to a certain probability range.
- So top p 1.0 means that the model considered all possible tokens.
- If you decrease the top p, for example, 0.5 limits, and this will limit the model to the top 50% of the most likely tokens, which can reduce the randomness and make the output more focused.
- So, for example, you can write the prompt as a write poem about the ocean, and if you provide the top p 1.0 will allow the more creative freedom Or if you decrease, the top p will result more precise and less varied output.

#### Frequency Penalty an Presence Penalty: 
- So basically frequency penalty is penalize the repeated words and presence penalties encourage the new topics.
- And you can balance repetition and creativity by changing these LLM settings.

### Function Calling in LLMs
- Function calling enables NLP to not just generate responses, but actually trigger external functions or APIs based on the prompt.
- In other words, the model isn't just giving you an answer.
- Instead of that, it is doing something for you.
- So this could include anything from retrieving live data, booking a meeting, or running a calculation.
- ![alt text](image-62.png)
- ![alt text](image-63.png)
- ![alt text](image-64.png)
- ![alt text](image-65.png)
- ![alt text](image-66.png)
- ![alt text](image-67.png)
- ![alt text](image-69.png)
- ![alt text](image-70.png)
- ![alt text](image-71.png)
- ![alt text](image-72.png)

## Small Language Models(SLMs)
- ![alt text](image-73.png)
- ![alt text](image-74.png)
- ![alt text](image-75.png)


## Exploring and Running LLMs
- ![alt text](image-76.png)
- ![alt text](image-78.png)
- ![alt text](image-79.png)
- ![alt text](image-80.png)
- ![alt text](image-81.png)
- ![alt text](image-82.png)
- ![alt text](image-83.png)
- ![alt text](image-84.png)
- ![alt text](image-85.png)
- ![alt text](image-86.png)
- ![alt text](image-87.png)
- ![alt text](image-88.png)
```python
from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": "Write a one-sentence bedtime story about a unicorn."
        }
    ]
)

print(completion.choices[0].message.content)

```
- Response is as follows:
```json
[
    {
        "index": 0,
        "message": {
            "role": "assistant",
            "content": "Under the soft glow of the moon, Luna the unicorn danced through fields of twinkling stardust, leaving trails of dreams for every child asleep.",
            "refusal": null
        },
        "logprobs": null,
        "finish_reason": "stop"
    }
]


```
- In addition to plain text, you can also have the model return structured data in JSON format - this feature is called Structured Outputs.
- You can provide instructions (prompts) to the model with differing levels of authority using message roles.
```python
 from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)

print(completion.choices[0].message.content)


```
- In .NET we can do it like this:
- We need to add a nuget package: dotnet add openai
```c#
using OpenAI.Chat;

ChatClient client = new(
  model: "gpt-4o", 
  apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")
);

ChatCompletion completion = client.CompleteChat("Say 'this is a test.'");

Console.WriteLine($"[ASSISTANT]: {completion.Content[0].Text}");


```
- If we want the response to be streamed i.e appear as it is being generated we can using OpenAI's CompleteChatStreamingAsync method:
```c#
using OpenAI.Chat;

ChatClient client = new(
  model: "gpt-4o-mini",
  apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")
);

string prompt = "Why is the sky blue?";

await foreach(var message in client.CompleteChatStreamingAsync(prompt))
{
    foreach(var item in message.ContentUpdate)
    {
        Console.Write(item.Text);
    }
}


```
- ![alt text](image-89.png)
- ![alt text](image-90.png)
- ![alt text](image-91.png)
- ![alt text](image-92.png)
- ![alt text](image-93.png)

### Interacting with Ollama Models using Code
- ![alt text](image-94.png)
- ![alt text](image-96.png)
- We need to install the following packages
```json
  <ItemGroup>
    <PackageReference Include="Microsoft.SemanticKernel" Version="1.44.0" />
    <PackageReference Include="Microsoft.SemanticKernel.Connectors.Ollama" Version="1.44.0-alpha" />
  </ItemGroup>

```
- Here is the program: 
```c#
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.ChatCompletion;

var builder = Kernel.CreateBuilder();
builder.AddOllamaChatCompletion("llama3.2:latest", new Uri("http://localhost:11434"));

var kernel = builder.Build();
var chatService = kernel.GetRequiredService<IChatCompletionService>();

var history = new ChatHistory();
history.AddSystemMessage("You are a helpful assistant");

while(true)
{
    Console.Write("You: ");
    var userMessage = Console.ReadLine();

    if(string.IsNullOrEmpty(userMessage))
    {
        break;
    }

    history.AddUserMessage(userMessage);

    var response = await chatService.GetChatMessageContentAsync(history);

    Console.WriteLine($"Bot: {response.Content}");

    history.AddMessage(response.Role, response.Content ?? string.Empty);
}


```
### Modernizing Modern Apps with AI-Powered LLM Capabilities
- ![alt text](image-97.png)
- ![alt text](image-98.png)
- ![alt text](image-99.png)
- ![alt text](image-100.png)
- ![alt text](image-101.png)
- ![alt text](image-102.png)
- ![alt text](image-103.png)

### LLM Augmentation Flow
- ![alt text](image-105.png)
- ![alt text](image-106.png)
- ![alt text](image-107.png)
- ![alt text](image-108.png)
- ![alt text](image-109.png)
- ![alt text](image-111.png)
- ![alt text](image-112.png)

## Prompt Engineering
- ![alt text](image-113.png)
- ![alt text](image-114.png)
- ![alt text](image-115.png)
- ![alt text](image-116.png)
- ![alt text](image-117.png)
- ![alt text](image-118.png)
- ![alt text](image-119.png)
- ![alt text](image-120.png)
- ![alt text](image-121.png)
- ![alt text](image-122.png)
- ![alt text](image-123.png)
- ![alt text](image-124.png)
- ![alt text](image-125.png)
- Better prompts get more accurate and relevant responses and allow us to shape the model's behavior
- Better prompts reduce vague or incorrect outputs by clarifying our instructions.
- Types of prompts: Zero Shot, Few shot and One-Shot
- ![alt text](image-126.png)

### Steps of Designing Effective Prompts
- ![alt text](image-127.png)
- ![alt text](image-128.png)
- ![alt text](image-129.png)
- ![alt text](image-130.png)
- ![alt text](image-131.png)
- ![alt text](image-132.png)
- ![alt text](image-133.png)
- ![alt text](image-134.png)
- ![alt text](image-135.png)
- Zero Shot prompting works well for general topics, but not for complex reasoning or specific data.
- ![alt text](image-136.png)
- ![alt text](image-137.png)
- ![alt text](image-138.png)
- ![alt text](image-139.png)
- ![alt text](image-140.png)
- ![alt text](image-141.png)
- ![alt text](image-142.png)
- ![alt text](image-143.png)
- ![alt text](image-144.png)
- ![alt text](image-145.png)
- ![alt text](image-146.png)
- Example of contextual prompting
- ![alt text](image-147.png)

### Design Advanced Prompts for EShop Support- Classification, Sentiment Analysis
- ![alt text](image-148.png)
- ![alt text](image-149.png)
- We can design an advanced prompt like this:
- You are an AI assistant integrated into an EShop support system. 
Your task is to help support agents by summarizing customer support interactions, classifying the ticket type, and evaluating customer sentiment based on the messages exchanged.

Here are the details of a customer support ticket:

- Product: {{product.Model}}
- Brand: {{product.Brand}}
- Ticket Messages: {{ticket.Messages}}

Please perform the following tasks:

1. **Summarization**: Write a detailed summary (up to 30 words) that includes:
    - The current status of the ticket.
    - Any specific questions asked by the customer.
    - What type of response would be most useful from the next support agent.
    - Avoid repeating the product or customer name unless necessary.

2. **Ticket Classification**: Based on the message log, classify the ticket into one of the following categories:
    - Question, Complaint, Feedback, Request for Refund, Product Issue, Other.
    - If the ticket contains multiple categories, choose the most dominant one.

3. **Customer Sentiment Analysis**: Analyze the latest message from the customer and determine their satisfaction level. Focus on the emotional tone of the customer, especially in how they express their needs or frustrations. Provide the satisfaction level using one of the following options: 
    - Very Dissatisfied, Dissatisfied, Neutral, Satisfied, Very Satisfied.

Return the output in this structured format (as JSON):
{
  "LongSummary": "string",
  "TicketClassification": "string",
  "CustomerSatisfaction": "string"
}

- ![alt text](image-150.png)
#### Prompts on Ticket Detail page can be designed as follows:
- Prompt: Q&A chat on the Ticket Detail Page with Retrieval-Augmented Generation and Citations

You are an AI assistant named 'Assistant' responsible for helping customer service agents handle support tickets for EShop specializing in electronics and computers.

The agent is currently handling the following ticket:

- Product: {{ProductId}}
- Customer: {{CustomerName}}
- Ticket summary: {{TicketSummary}}
- Customer's latest message: {{TicketLastCustomerMessage}}

You will be asked a question related to this ticket. When answering product-related questions, Always search the product manual or relevant documentation to ensure accuracy. 

**Citations** are critical in every response. After answering, provide a short, **verbatim** quote from the source to support your reply, using the following format:
- <cite source="manual/document_name">"Exact quote here" (max 10 words)</cite>.

Only include **one citation** per response, and ensure it is directly relevant to the question asked. Your responses should be clear, concise, and professional.


>>>>
Prompt 2: Q&A Chat Response Text Generator for Customer Communication on the Ticket Detail Page

You are an AI assistant helping a customer support team at EShop, and your task is to draft responses that agents can use to communicate with customers. Based on the customer’s latest message and overall sentiment, generate a suggested response that addresses the customer’s issue, provides helpful guidance, and maintains a friendly tone.

Here are the details:

- **Product**: {{product.Model}}
- **Ticket Summary**: {{request.TicketSummary}}
- **Customer's Latest Message**: {{request.TicketLastCustomerMessage}}

Analyze the sentiment of the customer's latest message and adjust the tone of the response accordingly:
- If the customer appears **frustrated**, include a sympathetic tone and offer reassurance.
- If the customer is **satisfied**, reinforce the positive experience and offer further assistance if needed.

Generate a response that meets the following guidelines:
- Address the customer's specific question or issue.
- Provide clear and concise instructions or solutions.
- Offer a friendly closing statement, inviting the customer to reach out if they need further help.

Return the response in this format:
{
  "Response": "string"
}


>>>>>
TEST TICKET DETAIL
>>
Prompt1: Q&A chat on the Ticket Detail Page with Retrieval-Augmented Generation and Citations
User Message: (show json viewer)
{
  "product": {
    "Model": "UltraView 4K Pro",
    "Brand": "VisionMax"
  },
  "request": {
    "TicketLastCustomerMessage": "Can you guide me on how to adjust the color settings for my UltraView 4K Pro? The colors seem off when I switch to HDMI mode."
  }
}

>>
Prompt 2: Q&A Chat Response Text Generator for Customer Communication on the Ticket Detail Page
User Message: (show json viewer) -- Customer Ticket 4: Customer Request for Return Policy
{
  "product": {
    "Model": "AirPro Max Blender",
    "Brand": "KitchenMaster"
  },
  "request": {
    "TicketSummary": "Customer inquiring about the return policy for a recently purchased blender.",
    "TicketLastCustomerMessage": "I bought the AirPro Max Blender two weeks ago, and it’s already malfunctioning. It overheats after just 30 seconds of use. Can I return it for a full refund?"
  }
}
- Provide the system instructions 
- ![alt text](image-151.png)
- ![alt text](image-152.png)
- ![alt text](image-153.png)
- ![alt text](image-154.png)
- ![alt text](image-155.png)
- ![alt text](image-156.png)


## Retrieval Augmented Generation(RAG)
- ![alt text](image-157.png)
- ![alt text](image-158.png)
- ![alt text](image-159.png)
- ![alt text](image-160.png)
- ![alt text](image-161.png)
- ![alt text](image-162.png)
- ![alt text](image-163.png)
- ![alt text](image-164.png)
- ![alt text](image-165.png)
- ![alt text](image-166.png)
- ![alt text](image-167.png)
- ![alt text](image-168.png)

### Ingestion with Embeddings and Vector Search
- ![alt text](image-170.png)
- ![alt text](image-171.png)
- ![alt text](image-172.png)
- ![alt text](image-173.png)
### Retrieval with ReRanking and Context Query Prompts
- ![alt text](image-174.png)
- ![alt text](image-175.png)
- ![alt text](image-176.png)
- ![alt text](image-177.png)
- ![alt text](image-178.png)
### Generation with Generator and Output 
- ![alt text](image-179.png)
- ![alt text](image-180.png)
- ![alt text](image-181.png)

### E2E Workflow of RAG
- ![alt text](image-182.png)
- ![alt text](image-183.png)
- ![alt text](image-184.png)
- ![alt text](image-185.png)
- ![alt text](image-187.png)
- ![alt text](image-188.png)

### Application Use Cases of RAG
- ![alt text](image-189.png)
- ![alt text](image-190.png)
- ![alt text](image-191.png)
- ![alt text](image-192.png)

