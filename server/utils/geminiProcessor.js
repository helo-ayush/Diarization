import { GoogleGenAI } from "@google/genai";

const genAI = new GoogleGenAI(process.env.GEMINI_API_KEY);


// Refine a raw diarized transcript using Gemini
// Fixes speaker assignments, cleans grammar, converts Hindi to Hinglish
export async function refineTranscript(rawTranscript, speakerCount) {

    const prompt = `You are a Senior Dialogue Editor. Your task is to transform a raw, messy diarized transcript into a clean, professional, and logically sound "Hinglish" conversation.

YOUR TASK:
1. **Analyze Context:** Read the entire transcript to understand the subject matter and identify the roles (e.g., Support Agent vs. Customer, Service Provider vs. Client).
2. **Reconstruct Logic:** Fix errors where the AI assigned words to the wrong person. Ensure the dialogue follows a logical flow of "Request -> Instruction -> Confirmation."
3. **Clean & Convert:** Fix grammar, remove stutters, and convert all Devanagari (Hindi) text into Romanized Hinglish.

═══════════════════════════════════
RAW TRANSCRIPT:
${rawTranscript}
═══════════════════════════════════

STRICT RULES:

📌 SPEAKER ASSIGNMENT:
- Identify the **Provider** (giving help/info) and the **User** (seeking help/info).
- If one person is giving a step-by-step process, all those steps belong to them—don't let the diarization split them.
- Short acknowledgments (e.g., "ji," "ok," "theek hai") belong to the person currently listening.
- Merge fragments: If a speaker's thought is split into three lines, combine them into one meaningful sentence.

📌 TEXT CLEANUP & HINGLISH:
- Convert ALL Devanagari script to Romanized Hinglish (e.g., "मैं देख सकता हूँ" -> "Main dekh sakta hoon").
- **Preserve Technical Terms:** Keep industry-specific English words as they are (e.g., "Server," "Refund," "Account," "Session," "Login").
- Remove repetitive filler words and fix broken grammar to make the conversation professional.

📌 OUTPUT FORMAT:
- Return ONLY a valid JSON array of objects.
- Each object must have "speaker" (integer) and "text" (string in Hinglish).
- Do NOT include any intro, outro, or markdown formatting. Just the JSON.

EXAMPLE OUTPUT:
[
  {"speaker": 0, "text": "Hello, main aapki kya madad kar sakta hoon?"},
  {"speaker": 1, "text": "Ji, mera account login nahi ho raha hai."}
]

IMPORTANT: The speaker count is ${speakerCount}. Ensure the final output is ONLY the JSON array.`;

    try {
        const startTime = Date.now();

        const response = await genAI.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: prompt,
            config: {
                temperature: 0.15,
                responseMimeType: "application/json",
            }
        });

        // Extract text from response
        let text = '';
        try {
            text = response.text?.trim() || '';
        } catch (e) {
            const candidates = response.candidates || [];
            if (candidates.length > 0) {
                const parts = candidates[0].content?.parts || [];
                for (const part of parts) {
                    if (part.text && !part.thought) {
                        text = part.text.trim();
                    }
                }
            }
        }

        const processingTime = Date.now() - startTime;

        if (!text) {
            console.warn('⚠️ Gemini returned empty response, using original transcript');
            return { transcript: null, processingTime };
        }

        console.log(`📝 Gemini response length: ${text.length} chars`);

        // Extract JSON array
        let jsonStr = text;
        const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (codeBlockMatch) {
            jsonStr = codeBlockMatch[1].trim();
        }

        const jsonMatch = jsonStr.match(/\[[\s\S]*\]/);
        if (!jsonMatch) {
            console.warn('⚠️ Gemini did not return valid JSON, using original transcript');
            console.warn('   First 500 chars of response:', text.substring(0, 500));
            return { transcript: null, processingTime };
        }

        const correctedSegments = JSON.parse(jsonMatch[0]);

        if (!Array.isArray(correctedSegments) || correctedSegments.length === 0) {
            console.warn('⚠️ Gemini returned empty or invalid array, using original transcript');
            return { transcript: null, processingTime };
        }

        // Validate structure
        const valid = correctedSegments.every(s =>
            typeof s.speaker === 'number' && typeof s.text === 'string' && s.text.trim().length > 0
        );

        if (!valid) {
            console.warn('⚠️ Gemini returned malformed segments, using original transcript');
            return { transcript: null, processingTime };
        }

        const refinedTranscript = correctedSegments
            .map(s => `Speaker ${s.speaker}: ${s.text}`)
            .join('\n');

        console.log(`🔧 Gemini produced ${correctedSegments.length} refined segments`);
        return { transcript: refinedTranscript, processingTime };

    } catch (err) {
        console.error('⚠️ Gemini refinement failed:', err.message);
        return { transcript: null, processingTime: 0 };
    }
}
