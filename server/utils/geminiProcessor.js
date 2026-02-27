import { GoogleGenAI } from "@google/genai";

const genAI = new GoogleGenAI(process.env.GEMINI_API_KEY);

export async function refineTranscript(rawTranscript, speakerCount) {
    const prompt = `You are a Senior Technical Dialogue Editor and Data Analyst. Your mission is to take a broken, raw diarized transcript and reconstruct it into a high-signal data object for a RAG-based search system.

═══════════════════════════════════
RAW TRANSCRIPT (High Error Rate):
${rawTranscript}
═══════════════════════════════════

YOUR WORKFLOW & RULES:

1. **Role Analysis & Labeling:**
   - Determine who is the 'Agent' (Provider) and who is the 'Client' (User). 
   - Replace "Speaker 0/1" with these labels consistently.

2. **Vector-Optimized "High-Contrast" Summary:**
   - Write a summary specifically for an embedding vector.
   - **Include Contrast:** Explicitly state the technical gap (e.g., "Client struggled with X in the ERP, Agent resolved it using Y").
   - Mention all technical entities (AnyDesk, Session years, Button names). This ensures similarity searches for "AnyDesk" or "2026 session" hit this record.

3. **Smart Transcript Reconstruction (THE DEEP CLEAN):**
   - **Prioritize Meaning:** The raw text is often unmeaningful. Your job is to make it meaningful. If the diarization is wrong, swap the speakers.
   - **Rewrite for Clarity:** If a sentence is a mess of fragments, rewrite it into a complete, professional sentence that captures the *intent*.
   - **Merge Thoughts:** Join consecutive lines from the same role into one cohesive paragraph.
   - **Remove Noise:** Delete stutters and "ok ok ok" repetitions, but keep the "logic" of the confirmation.

4. **Hinglish & Formatting:**
   - Convert all Devanagari (Hindi) to Romanized Hinglish.
   - Keep technical English words exactly as they are.

5. **Satisfaction Score:**
   - Rate the Client's satisfaction from 1-10 based on the resolution and their final tone.

STRICT JSON OUTPUT FORMAT:
{
  "summary": "Dense, technical, high-contrast summary for embeddings...",
  "satisfactionScore": 10,
  "detectedRoles": { "speaker0": "Agent", "speaker1": "Client" },
  "tags": ["ERP", "Session Migration", "Technical Support"], 
  "refinedTranscript": [
    { "role": "Agent", "text": "Hinglish reconstructed text..." },
    { "role": "Client", "text": "Hinglish reconstructed text..." }
  ]
}

IMPORTANT: Return ONLY the JSON. Do not add any conversational text. Use the speaker count (${speakerCount}) to guide your role detection.`;

    try {
        const startTime = Date.now();

        const result = await genAI.models.generateContent({
            model: "gemini-3-flash-preview",
            contents: prompt,
            config: {
                temperature: 0.1,
                responseMimeType: "application/json",
            }
        });

        const data = JSON.parse(result.text);
        const processingTime = Date.now() - startTime;

        // Joining for your database/view
        const finalTranscriptString = data.refinedTranscript
            .map(s => `${s.role}: ${s.text}`)
            .join('\n');
        return {
            summary: data.summary,
            satisfactionScore: data.satisfactionScore,
            tags: data.tags,
            transcript: finalTranscriptString, // The "Agent: text" format
            processingTime
        };

    } catch (err) {
        console.error('⚠️ Processing failed:', err.message);
        return {
            transcript: null,
            summary: "Could not generate summary",
            satisfactionScore: 0,
            tags: [],
            processingTime: 0
        };
    }
}