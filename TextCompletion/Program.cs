using OpenAI.Chat;

ChatClient client = new(
  model: "gpt-4o-mini",
  apiKey: Environment.GetEnvironmentVariable("OPENAI_API_KEY")
);

string prompt = "Classify the following text: 'The sky appears blue due to a phenomenon called Rayleigh scattering. When sunlight enters the Earth's atmosphere, it interacts with air molecules and small particles. Sunlight, or white light, is made up of many colors, each with different wavelengths. Blue light has a shorter wavelength compared to other colors like red or yellow.'";

await foreach(var message in client.CompleteChatStreamingAsync(prompt))
{
    foreach(var item in message.ContentUpdate)
    {
        Console.Write(item.Text);
    }
}