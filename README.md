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
- 