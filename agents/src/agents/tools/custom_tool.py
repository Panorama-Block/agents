from crewai.tools import BaseTool
from google import genai
from google.genai import types
import os
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import openai
import logging
from typing import Any
import time
from datetime import datetime

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
grok_api_key = os.getenv("GROK_API_KEY")
agent = os.getenv("PROCCESS_TYPE")

client = genai.Client(api_key=api_key)

logger = logging.getLogger(__name__)

class GeminiImageDirectTool(BaseTool):
    name: str = "generate_image_direct"
    description: str = "Generate images using Gemini Pro model based on the prompt"

    def _run(self, prompt: str, num_images: int = 1) -> str:
        try:
            # enhanced_prompt = f"""
            # Create a detailed image based on this description:
            # {prompt}
            
            # Please provide a detailed description of the image that would be generated.
            # Focus on visual elements, composition, colors, and style.
            # """
            
            response = client.models.generate_images(
                model='imagen-3.0-generate-002',
                prompt=prompt,
                config=types.GenerateImagesConfig(
                    number_of_images= num_images,
                )
            )
            for generated_image in response.generated_images:
                image = Image.open(BytesIO(generated_image.image.image_bytes))
                image.save('image.png')
                image.show()
                
            return 'image.png'
        except Exception as e:
            return f"Erro ao gerar a imagem: {e}"

class GrokSearchTool(BaseTool):
    name: str = "grok_search"
    description: str = "Search for content using Grok API"
    client: Any = None  
    failure_count: int = 0
    last_failure_time: Any = None

    search_type: str = ""

    if agent == "zico":
        search_type = "Zico"
    elif agent == "avax":
        search_type = "Avalanche (AVAX)"
    elif agent == "hedera":
        search_type = "Hedera (HBAR)"

    def __init__(self):
        super().__init__()
        self.client = openai.OpenAI(
            api_key=os.getenv("GROK_API_KEY"),
            base_url="https://api.x.ai/v1",
            timeout=30.0
        )

    def _run(self, query: str) -> str:
        """
        Execute a search using Grok API
        Args:
            query (str): The search query
        Returns:
            str: The search results
        """
        max_retries = 3
        
        if self.failure_count > 5:
            cooldown = min(60, self.failure_count * 5)
            logger.warning(f"Too many failures ({self.failure_count}), cooling down for {cooldown}s")
            time.sleep(cooldown)
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Executing Grok search attempt {attempt+1}/{max_retries}: {query}")
                completion = self.client.chat.completions.create(
                    model="grok-3-beta",
                    messages=[
                        {
                            "role": "system", 
                            "content": """You are a research assistant focused on {self.search_type}. 
                            Search and analyze only the specific information requested.
                            Provide factual, data-driven insights based on real-time information.
                            Keep responses focused and relevant to the query.
                            If you cannot find relevant information, explain why."""
                        },
                        {
                            "role": "user", 
                            "content": f"Search and provide specific information about: {query}\nFocus only on recent and verified information about this topic in the context of {self.search_type}."
                        }
                    ],
                    max_tokens=500,
                    temperature=0.7
                )
                
                content = completion.choices[0].message.content
                if not content or content.strip() == "":
                    self.failure_count += 1
                    if attempt < max_retries - 1:
                        logger.warning(f"Empty response received on attempt {attempt + 1}, retrying...")
                        time.sleep(2 * (attempt + 1))
                        continue
                    return "No relevant information found. Please try a different query or check back later."
                
                self.failure_count = 0
                self.last_failure_time = None
                return content

            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = datetime.now()
                logger.error(f"Error executing Grok search (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(3 * (attempt + 1))
                    continue
                return f"Error executing Grok search after {max_retries} attempts: {str(e)}"

    async def _arun(self, query: str) -> str:
        """Async implementation of the tool"""
        return self._run(query)