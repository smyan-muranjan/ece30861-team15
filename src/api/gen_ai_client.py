import json
import logging
import os
import re
import ssl

import aiohttp


class GenAIClient:
    def __init__(self):
        self.url = "https://genai.rcac.purdue.edu/api/chat/completions"
        env_api_key = os.environ.get("GENAI_API_KEY")
        self.has_api_key = bool(env_api_key)
        if env_api_key:
            self.headers = {
                "Authorization": f"Bearer {env_api_key}",
                "Content-Type": "application/json"
            }
        else:
            self.headers = {
                "Content-Type": "application/json"
            }

    def preprocess_readme_for_analysis(self, readme_text: str, max_chars: int = 3000) -> str:
        """Extract relevant sections from README and limit size for reliable processing."""
        if not readme_text:
            return ""
        
        # If README is small enough, return as-is
        if len(readme_text) <= max_chars:
            return readme_text
        
        # Extract key sections that are most relevant for analysis
        sections = []
        lines = readme_text.split('\n')
        
        # Keywords for performance-related content
        perf_keywords = ['performance', 'benchmark', 'evaluation', 'results', 'metrics', 
                        'accuracy', 'score', 'f1', 'precision', 'recall', 'bleu', 'rouge',
                        'evaluation', 'test', 'dataset', 'model', 'experiment']
        
        # Keywords for documentation quality
        doc_keywords = ['installation', 'install', 'setup', 'usage', 'example', 'quickstart',
                       'getting started', 'tutorial', 'guide', 'documentation', 'api',
                       'requirements', 'dependencies', 'configuration']
        
        current_section = []
        capture_mode = None
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if this is a header line
            is_header = line.strip().startswith('#') or (
                i + 1 < len(lines) and 
                lines[i + 1].strip() and 
                all(c in '=-' for c in lines[i + 1].strip())
            )
            
            # Start capturing performance sections
            if is_header and any(keyword in line_lower for keyword in perf_keywords):
                if current_section and capture_mode:
                    sections.append(('\n'.join(current_section), capture_mode))
                current_section = [line]
                capture_mode = 'performance'
            
            # Start capturing documentation sections
            elif is_header and any(keyword in line_lower for keyword in doc_keywords):
                if current_section and capture_mode:
                    sections.append(('\n'.join(current_section), capture_mode))
                current_section = [line]
                capture_mode = 'documentation'
            
            # Stop capturing when hitting unrelated sections
            elif is_header and capture_mode and not any(
                keyword in line_lower for keyword in perf_keywords + doc_keywords
            ):
                if current_section:
                    sections.append(('\n'.join(current_section), capture_mode))
                current_section = []
                capture_mode = None
            
            # Add lines to current section if capturing
            elif capture_mode:
                current_section.append(line)
                
                # Stop if section gets too long
                if len('\n'.join(current_section)) > max_chars // 2:
                    sections.append(('\n'.join(current_section), capture_mode))
                    current_section = []
                    capture_mode = None
        
        # Add the last section if we were capturing
        if current_section and capture_mode:
            sections.append(('\n'.join(current_section), capture_mode))
        
        # If no relevant sections found, take strategic parts of the README
        if not sections:
            # Take the beginning (usually has overview/description)
            first_part = readme_text[:max_chars // 2]
            
            # Look for any numbers/metrics in the middle portion
            middle_start = len(readme_text) // 3
            middle_end = min(middle_start + max_chars // 2, len(readme_text))
            middle_part = readme_text[middle_start:middle_end]
            
            # Look for numbers, percentages, or metric-like patterns
            if re.search(r'\d+\.?\d*\s*%|accuracy|f1|score|benchmark', middle_part.lower()):
                return first_part + "\n\n[...middle section...]\n\n" + middle_part
            else:
                return first_part
        
        # Prioritize performance sections, then documentation
        performance_sections = [s[0] for s in sections if s[1] == 'performance']
        doc_sections = [s[0] for s in sections if s[1] == 'documentation']
        
        # Combine sections within character limit
        combined = ""
        
        # Add performance sections first (higher priority)
        for section in performance_sections:
            if len(combined + section) <= max_chars:
                combined += section + "\n\n"
            else:
                # Add partial section if it fits
                remaining = max_chars - len(combined) - 10
                if remaining > 100:
                    combined += section[:remaining] + "...\n\n"
                break
        
        # Add documentation sections if space remains
        for section in doc_sections:
            if len(combined + section) <= max_chars:
                combined += section + "\n\n"
            else:
                # Add partial section if it fits
                remaining = max_chars - len(combined) - 10
                if remaining > 100:
                    combined += section[:remaining] + "..."
                break
        
        # If still no content, take the first part of the README
        if not combined.strip():
            combined = readme_text[:max_chars]
        
        return combined.strip()

    async def chat(self, message: str, model: str = "llama3.3:70b", system_prompt: str = None) -> str:
        # If no API key is available, return a default response
        if not self.has_api_key:
            return "No performance claims found in the documentation."

        messages = []
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": message
        })

        body = {
            "model": model,
            "messages": messages
        }
        
        # Calculate total payload size
        total_chars = sum(len(msg["content"]) for msg in messages)
        print(f"      📤 Sending {total_chars:,} characters to {model}")
        
        # Create SSL context that doesn't verify certificates for servers
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        connector = aiohttp.TCPConnector(ssl=ssl_context)
        async with aiohttp.ClientSession(connector=connector) as session:
            async with session.post(
                self.url, headers=self.headers, json=body
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    content = data['choices'][0]['message']['content']
                    print(f"      📥 Received {len(content)} characters from API")
                    return content
                elif response.status == 401:
                    print(f"      🔒 Authentication failed (401)")
                    # Authentication failed - return default response
                    return "No performance claims found in the documentation."
                else:
                    error = await response.text()
                    print(f"      ❌ API Error {response.status}: {error[:100]}...")
                    raise Exception(f"Error: {response.status}, {error}")

    def _fallback_performance_parsing(self, readme_text: str) -> dict:
        """Fallback method using keyword matching when LLM fails."""
        readme_lower = readme_text.lower()
        
        # Check for quantitative metrics keywords
        metric_keywords = [
            'accuracy', 'f1-score', 'f1 score', 'precision', 'recall', 'auc', 'map',
            'bleu', 'rouge', 'meteor', 'cider', 'spice', 'bert-score', 'perplexity',
            'loss', 'error rate', 'top-1', 'top-5', 'mse', 'mae', 'r2', 'correlation',
            '%', 'percent', 'score', 'metric'
        ]
        
        # Look for numbers with these keywords
        has_metrics = False
        for keyword in metric_keywords:
            # Look for patterns like "accuracy: 94.2%" or "F1-score of 0.89"
            pattern = rf'{keyword}[\s:=]+\d+\.?\d*\s*%?'
            if re.search(pattern, readme_lower):
                has_metrics = True
                break
        
        # Also check for standalone percentages or decimal scores
        if not has_metrics:
            perf_patterns = [
                r'\d+\.?\d*\s*%',  # Percentages like "94.2%"
                r'[=:]\s*0\.\d+',  # Decimal scores like "= 0.94"
                r'\d+\.?\d*\s*(accuracy|precision|recall|f1)',  # "94.2 accuracy"
            ]
            for pattern in perf_patterns:
                if re.search(pattern, readme_lower):
                    has_metrics = True
                    break
        
        # Check for standard benchmark keywords
        benchmark_keywords = [
            'glue', 'superglue', 'imagenet', 'coco', 'squad', 'squad2.0', 'squad 2.0',
            'librispeech', 'commonvoice', 'wmt', 'bleu', 'rouge', 'meteor',
            'benchmark', 'evaluation', 'dataset', 'test set', 'validation set',
            'leaderboard', 'competition', 'challenge', 'sota', 'state-of-the-art'
        ]
        
        mentions_benchmarks = any(keyword in readme_lower for keyword in benchmark_keywords)
        
        return {
            "has_metrics": 1 if has_metrics else 0,
            "mentions_benchmarks": 1 if mentions_benchmarks else 0
        }

    def _fallback_clarity_assessment(self, readme_text: str) -> float:
        """Keyword-based fallback when LLM parsing fails."""
        if not readme_text or len(readme_text.strip()) < 50:
            return 0.0
        
        score = 0.0
        readme_lower = readme_text.lower()
        
        # Check for key documentation elements (each worth 0.2 points)
        clarity_indicators = [
            # Installation/setup information
            (['install', 'installation', 'setup', 'getting started', 'quickstart'], 0.2),
            # Usage information
            (['usage', 'example', 'how to', 'tutorial', 'guide'], 0.2),
            # Requirements/dependencies
            (['requirements', 'dependencies', 'prerequisite'], 0.15),
            # API documentation
            (['api', 'function', 'method', 'parameter', 'argument'], 0.15),
            # Clear structure (headers)
            (['#', '##', '###'], 0.1),
            # Code examples
            (['```', 'python', 'import', 'def ', 'class '], 0.1),
            # Contact/support info
            (['contact', 'support', 'issue', 'bug', 'contribute'], 0.1),
        ]
        
        for keywords, points in clarity_indicators:
            if any(keyword in readme_lower for keyword in keywords):
                score += points
        
        # Length bonus (well-documented projects tend to be longer)
        if len(readme_text) > 1000:
            score += 0.1
        if len(readme_text) > 3000:
            score += 0.1
        
        # Penalty for very short READMEs
        if len(readme_text) < 200:
            score *= 0.5
        
        return min(1.0, score)

    def _extract_json_from_response(self, response: str) -> dict:
        """Extract JSON from LLM response with multiple strategies."""
        # Strategy 1: Look for JSON blocks in markdown
        json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
        for block in json_blocks:
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
        
        # Strategy 2: Look for standalone JSON objects
        json_matches = re.findall(r'\{[^{}]*"has_metrics"[^{}]*"mentions_benchmarks"[^{}]*\}', response)
        for match in json_matches:
            try:
                parsed = json.loads(match)
                if "has_metrics" in parsed and "mentions_benchmarks" in parsed:
                    return {
                        "has_metrics": int(parsed["has_metrics"]),
                        "mentions_benchmarks": int(parsed["mentions_benchmarks"])
                    }
            except (json.JSONDecodeError, ValueError, KeyError):
                continue
        
        # Strategy 3: Look for any JSON-like structure
        json_pattern = r'\{[^{}]*\}'
        matches = re.findall(json_pattern, response)
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict):
                    # Try to map to expected format
                    has_metrics = 0
                    mentions_benchmarks = 0
                    
                    for key, value in parsed.items():
                        key_lower = key.lower()
                        if 'metric' in key_lower or 'performance' in key_lower:
                            has_metrics = int(bool(value))
                        elif 'benchmark' in key_lower or 'evaluation' in key_lower:
                            mentions_benchmarks = int(bool(value))
                    
                    return {
                        "has_metrics": has_metrics,
                        "mentions_benchmarks": mentions_benchmarks
                    }
            except (json.JSONDecodeError, ValueError):
                continue
        
        # If all parsing fails, return None to trigger fallback
        return None

    def _extract_float_from_response(self, response: str) -> float:
        """Extract float from LLM response with multiple strategies."""
        # Strategy 1: Direct float parsing
        try:
            cleaned = response.strip()
            # Remove common text prefixes
            for prefix in ["score:", "clarity:", "rating:", "the score is", "result:"]:
                if cleaned.lower().startswith(prefix):
                    cleaned = cleaned[len(prefix):].strip()
            
            if cleaned.replace('.', '').replace('0', '').replace('1', '').replace('2', '').replace('3', '').replace('4', '').replace('5', '').replace('6', '').replace('7', '').replace('8', '').replace('9', '').strip() == '':
                score = float(cleaned)
                return max(0.0, min(1.0, score))
        except ValueError:
            pass
        
        # Strategy 2: Extract first valid decimal number
        decimal_matches = re.findall(r'\b(?:0\.\d+|1\.0+|0\.0+|1)\b', response)
        for match in decimal_matches:
            try:
                score = float(match)
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        
        # Strategy 3: Look for percentage and convert
        percent_matches = re.findall(r'(\d+\.?\d*)\s*%', response)
        for match in percent_matches:
            try:
                score = float(match) / 100.0
                if 0.0 <= score <= 1.0:
                    return score
            except ValueError:
                continue
        
        # Strategy 4: Look for any number and normalize
        number_matches = re.findall(r'\d+\.?\d*', response)
        for match in number_matches:
            try:
                score = float(match)
                # If it's a reasonable score, use it
                if 0.0 <= score <= 1.0:
                    return score
                elif 1.0 < score <= 10.0:  # Scale like 1-10 to 0-1
                    return score / 10.0
                elif 10.0 < score <= 100.0:  # Scale like 1-100 to 0-1
                    return score / 100.0
            except ValueError:
                continue
        
        # If all parsing fails, return None to trigger fallback
        return None

    async def get_performance_claims(self, readme_text: str) -> dict:
        # If no API key is available, return fallback result
        if not self.has_api_key:
            print("   🔑 No API key - using fallback parsing")
            return {"has_metrics": 0, "mentions_benchmarks": 0}

        # Preprocess README to manageable size
        processed_readme = self.preprocess_readme_for_analysis(readme_text, max_chars=3000)
        print(f"   📝 README: {len(readme_text):,} chars → {len(processed_readme):,} chars processed")
        print(f"   🌐 Making Performance Claims API call...")
        
        # Load system prompt from file
        with open("src/api/performance_claims_system_prompt.txt", "r") as f:
            system_prompt = f.read().strip()
        
        # Load user prompt template from file
        with open("src/api/performance_claims_user_prompt.txt", "r") as f:
            user_prompt_template = f.read().strip()
        
        # Format the user prompt with the processed README
        user_prompt = user_prompt_template.format(processed_readme=processed_readme)

        try:
            response = await self.chat(user_prompt, system_prompt=system_prompt)
            print(f"   ✅ API Response: {len(response)} chars - {response[:100]}...")
            
            # Try to extract JSON using robust parsing
            result = self._extract_json_from_response(response)
            
            if result is not None:
                # Validate the result
                has_metrics = result.get("has_metrics", 0)
                mentions_benchmarks = result.get("mentions_benchmarks", 0)
                
                if has_metrics in [0, 1] and mentions_benchmarks in [0, 1]:
                    print(f"   ✅ Parsed result: {result}")
                    return result
                else:
                    print(f"   ⚠️  Invalid values in LLM response: {result}")
                    logging.warning(f"Invalid values in LLM response: {result}")
            
            # If JSON parsing failed, use fallback
            print(f"   🔄 JSON parsing failed, using fallback parsing")
            logging.warning(f"JSON parsing failed, using fallback for response: {response[:200]}...")
            return self._fallback_performance_parsing(processed_readme)
            
        except Exception as e:
            print(f"   ❌ API call failed: {e}")
            logging.warning(f"Performance claims analysis failed: {e}")
            # Use fallback parsing
            return self._fallback_performance_parsing(processed_readme)

    async def get_readme_clarity(self, readme_text: str) -> float:
        # If no API key is available, return fallback score
        if not self.has_api_key:
            print("   🔑 No API key - using fallback assessment")
            return self._fallback_clarity_assessment(readme_text)

        # Preprocess README - limit to first 2000 characters for clarity assessment
        processed_readme = readme_text[:2000] if readme_text else ""
        print(f"   📝 README: {len(readme_text):,} chars → {len(processed_readme):,} chars processed")
        print(f"   🌐 Making README Clarity API call...")
        
        # Load system prompt from file
        with open("src/api/readme_clarity_system_prompt.txt", "r") as f:
            system_prompt = f.read().strip()
        
        # Load user prompt template from file
        with open("src/api/readme_clarity_user_prompt.txt", "r") as f:
            user_prompt_template = f.read().strip()
        
        # Format the user prompt with the processed README
        user_prompt = user_prompt_template.format(processed_readme=processed_readme)

        try:
            response = await self.chat(user_prompt, system_prompt=system_prompt)
            print(f"   ✅ API Response: {len(response)} chars - '{response.strip()}'")
            
            # Extract score using robust parsing
            score = self._extract_float_from_response(response)
            
            if score is not None:
                print(f"   ✅ Parsed score: {score}")
                return score
            
            # If parsing failed, use fallback
            print(f"   🔄 Float parsing failed, using fallback assessment")
            logging.warning(f"Float parsing failed, using fallback for response: {response[:100]}...")
            return self._fallback_clarity_assessment(processed_readme)
            
        except Exception as e:
            print(f"   ❌ API call failed: {e}")
            logging.warning(f"README clarity analysis failed: {e}")
            # Use fallback assessment
            return self._fallback_clarity_assessment(processed_readme)


if __name__ == "__main__":
    import asyncio

    async def main():
        for i in range(1, 6):
            start_time = asyncio.get_event_loop().time()
            client = GenAIClient()
            contents = ""
            with open('testREADME.md', 'r', encoding='utf-8') as f:
                contents = f.read()
            await client.chat("summarize this: " + contents[:i*1000])
            end_time = asyncio.get_event_loop().time()
            print(f"API call took {end_time - start_time:.2f} seconds")

    asyncio.run(main())
