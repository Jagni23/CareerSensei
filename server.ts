import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI, Type } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Set up JSON parsers with generous limits for file uploads
app.use(express.json({ limit: "25mb" }));
app.use(express.urlencoded({ limit: "25mb", extended: true }));

// Lazy init of Gemini Client
let aiClient: GoogleGenAI | null = null;
function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn("Waring: GEMINI_API_KEY is not defined in environment variables. Calls will fail.");
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey || "MOCK_KEY",
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// Ensure error handling doesn't crash the server
const executeQueryPrompt = async (prompt: string, schema?: any, systemInstruction?: string) => {
  const ai = getGeminiClient();
  const config: any = {};
  
  if (systemInstruction) {
    config.systemInstruction = systemInstruction;
  }
  
  if (schema) {
    config.responseMimeType = "application/json";
    config.responseSchema = schema;
  }

  const response = await ai.models.generateContent({
    model: "gemini-3.5-flash",
    contents: prompt,
    config,
  });

  return response.text;
};

// 1. Analyze Resume Endpoint
// Supports Base64 file (PDF or Text) or raw pasted text
app.post("/api/analyze-resume", async (req, res) => {
  try {
    const { fileBase64, mimeType, rawText, sampleName } = req.body;
    let resumeContentPrompt = "";

    // If a sample is requested, generate standard realistic high-profile text first
    if (sampleName) {
      if (sampleName === "frontend") {
        resumeContentPrompt = `Name: Sarah Jenkins
Role: Frontend Developer / UI Specialist
Experience: 3 Years
Email: sarah.jenkins@example.com
About: Self-taught react engineer with a background in design, lover of micro-interactions and high-performance frontend visual assets.
Skills: HTML, CSS, JavaScript (ES6+), React 18, Tailwind CSS, TypeScript, Vite, Framer Motion, Redux, Git.
Experience:
- Frontend Engineer at PixelLabs (2024-Present): Designed and developed consumer web portals. Streamlined web assets resulting in a 40% performance enhancement. Implemented standard custom-designed component widgets with Tailwind and motion.
- Junior UI Web Developer at CraftyStudio (2023-2024): Collaborated closely with product designers to implement beautiful responsive interfaces. Re-engineered legacy layout into modern CSS Flex/Grid.
Education:
- B.S. in Communication & Media Design (2022)
Projects:
- MotionUI Library: Open-source utility containing visual interactive micro-components with custom physics. Received 500+ GitHub stars.
- SlateEditor: Minimalist offline-first markdown preview writer utilizing local persistence.
Certifications:
- Certified Scrum Developer (2024)
- React Advanced Engineering Nanodegree (2023)`;
      } else if (sampleName === "ai-engineer") {
        resumeContentPrompt = `Name: Marcus Chen
Role: Junior AI Implementation Specialist
Experience: 2 Years
Email: marcus.chen@example.com
About: Passionate machine learning developer focused on integrating commercial LLM architectures and prompt orchestration.
Skills: Python, PyTorch, SQL, HuggingFace, OpenAI API, LangChain, Node.js, Next.js, API Integration, Docker.
Experience:
- Software Engineer (AI integrations) at SmartFlow Solutions (2024-Present): Built client-facing generative features utilizing modern orchestrator SDKs. Reduced token overhead by 30% through engineered system prompts and response caching. Exposed server-side API endpoints for live web streams.
- Machine Learning Intern at Databot AI (2023-2024): Evaluated accuracy of retrieval frameworks. Pre-processed datasets of 1M+ technical documents for model alignment.
Certifications:
- DeepLearning.AI Generative AI for Everyone (2024)
- AWS Certified Cloud Practitioner (2023)
Projects:
- InsightPDF: Chat-with-document app that parses PDF documents server-side and indexes vectors into low-latency memory store.
- AutoScrape Insights: Python analytics pipeline digesting stock tickers and summarizing market sentiments hourly.`;
      } else {
        resumeContentPrompt = `Name: Alex Sterling
Role: General Software Engineer / Full Stack Developer
Experience: 4 Years
Skills: JavaScript, Python, Node.js, Express, PostgreSQL, React, AWS, Docker, REST APIs, TypeScript, Git
Experience:
- Full-Stack Developer at NexaCore Technologies (2022-Present): Designed database schemas and optimized nested queries. Structured secure authentication routes. Mentored two junior engineers.
- Software Intern at SysSystems (2021-2022): Developed dashboard displays and maintained internal reporting scripts. Certified database migration workflows.`;
      }
    } else if (fileBase64 && mimeType) {
      if (mimeType.includes("pdf")) {
        // Send base64 PDF directly to Gemini!
        const ai = getGeminiClient();
        const response = await ai.models.generateContent({
          model: "gemini-3.5-flash",
          contents: [
            {
              inlineData: {
                data: fileBase64,
                mimeType: "application/pdf"
              }
            },
            {
              text: "Extract and summarize this resume's information. Give me a raw, comprehensive plain-text dump detailing: Full Name, Role Title, Email, About summary, Skills, Professional Experience, Education, Projects, and Certifications."
            }
          ]
        });
        resumeContentPrompt = response.text || "Failed to extract clear text from the PDF.";
      } else {
        // Assume text-based file
        const decodedText = Buffer.from(fileBase64, "base64").toString("utf-8");
        resumeContentPrompt = decodedText;
      }
    } else if (rawText) {
      resumeContentPrompt = rawText;
    } else {
      return res.status(400).json({ error: "Missing resume files, text input or sample option." });
    }

    // Now, run the Structured Extraction call
    const extractionSchema = {
      type: Type.OBJECT,
      properties: {
        name: { type: Type.STRING, description: "Extract the candidate's full name" },
        title: { type: Type.STRING, description: "Extract overall professional title or target role" },
        email: { type: Type.STRING, description: "Candidate email" },
        about: { type: Type.STRING, description: "A summary profile of the candidate" },
        skills: { 
          type: Type.ARRAY, 
          items: { type: Type.STRING }, 
          description: "List of precise tools, languages, frameworks, or soft skills" 
        },
        experience: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              role: { type: Type.STRING },
              company: { type: Type.STRING },
              period: { type: Type.STRING },
              responsibilities: { type: Type.ARRAY, items: { type: Type.STRING } }
            },
            required: ["role", "company", "period", "responsibilities"]
          }
        },
        education: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Schools, degrees, fields of study"
        },
        projects: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              projectName: { type: Type.STRING },
              description: { type: Type.STRING }
            },
            required: ["projectName", "description"]
          }
        },
        certifications: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Certifications earned"
        },
        // CareerSensei Metrics Calculated Dynamically
        marketReadinessScore: { 
          type: Type.INTEGER, 
          description: "Dynamic score out of 100 on how job-market ready the resume is." 
        },
        marketReadinessBreakdown: {
          type: Type.OBJECT,
          properties: {
            resumeQuality: { type: Type.INTEGER, description: "Score out of 100 for formatting, typography, wording" },
            skillsAlignment: { type: Type.INTEGER, description: "Score out of 100 for standard in-demand technologies" },
            projectImpact: { type: Type.INTEGER, description: "Score out of 100 for tangible engineered projects" },
            certificationsStrength: { type: Type.INTEGER, description: "Score out of 100 for accredited specialized credentials" }
          },
          required: ["resumeQuality", "skillsAlignment", "projectImpact", "certificationsStrength"]
        },
        atsCompatibilityScore: { 
          type: Type.INTEGER, 
          description: "Calculated ATS parsed compatibility percentage (e.g. 70-98)." 
        },
        atsCompatibilityExplanation: {
          type: Type.STRING,
          description: "Short analysis of how easy it is for an automated parser to understand this formatting and word layout."
        },
        industryDemand: { 
          type: Type.STRING, 
          description: "Industry hiring demand rating (e.g. HIGH, MEDIUM, COLD)" 
        },
        industryDetail: {
          type: Type.STRING,
          description: "One short power paragraph evaluating active hiring trends in this candidate's space."
        },
        sentimentToneAnalysis: {
          type: Type.STRING,
          description: "Analyze the tone: Is it highly passive (focused on duty declarations) or highly visionary/impact-driven (focused on metric outcomes)?"
        },
        salaryBenchmarkMin: { type: Type.INTEGER, description: "Estimated realistic entry-mid annual salary bottom end (e.g., 75000)" },
        salaryBenchmarkMax: { type: Type.INTEGER, description: "Estimated realistic annual salary top end (e.g., 125000)" },
      },
      required: [
        "name", "title", "email", "about", "skills", "experience", "education", 
        "projects", "certifications", "marketReadinessScore", "marketReadinessBreakdown",
        "atsCompatibilityScore", "atsCompatibilityExplanation", "industryDemand", "industryDetail",
        "sentimentToneAnalysis", "salaryBenchmarkMin", "salaryBenchmarkMax"
      ]
    };

    const sysMsg = `You are the CareerSensei Core Analyzer. Parse the candidate resume text, extract accurate details, and compute professional, non-fluffy market readiness metrics, ATS compatibility estimations, industry demand (from high, medium, to cold), salary benchmarking estimates, and a sentiment tone review. Ensure stats are dynamically balanced according to the actual strength of the extracted resume content. Return strictly valid JSON formatted to match the provided schema.`;

    const resultText = await executeQueryPrompt(
      `Please extract information and analyze: \n\n${resumeContentPrompt}`,
      extractionSchema,
      sysMsg
    );

    if (!resultText) {
      throw new Error("Empty analysis result from model.");
    }

    const payload = JSON.parse(resultText);
    payload.extractedRawText = resumeContentPrompt; // save plain text for further context
    res.json(payload);
  } catch (error: any) {
    console.error("Error analyzing resume:", error);
    res.status(500).json({ error: error.message || "Failed to analyze resume details." });
  }
});

// 2. Target Gap Analysis Endpoint
app.post("/api/gap-analysis", async (req, res) => {
  try {
    const { profile, targetRole } = req.body;
    if (!profile || !targetRole) {
      return res.status(400).json({ error: "Missing candidate profile or target role specification." });
    }

    const gapSchema = {
      type: Type.OBJECT,
      properties: {
        targetRoleAnalyzed: { type: Type.STRING },
        overallGapPercentage: { type: Type.INTEGER, description: "estimated gap out of 100%" },
        missingSkills: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              skill: { type: Type.STRING },
              importance: { type: Type.STRING, description: "CRITICAL, HIGH, or BENEFICIAL" },
              relevanceExplanation: { type: Type.STRING }
            },
            required: ["skill", "importance", "relevanceExplanation"]
          }
        },
        missingToolsAndPlatforms: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Specific libraries, deployment environments or tools that are missing"
        },
        missingCertifications: {
          type: Type.ARRAY,
          items: {
            type: Type.OBJECT,
            properties: {
              name: { type: Type.STRING },
              provider: { type: Type.STRING },
              valueProposition: { type: Type.STRING }
            },
            required: ["name", "provider", "valueProposition"]
          }
        },
        recommendedProjectIdea: {
          type: Type.OBJECT,
          properties: {
            title: { type: Type.STRING },
            techStack: { type: Type.ARRAY, items: { type: Type.STRING } },
            concept: { type: Type.STRING },
            keyFeatures: { type: Type.ARRAY, items: { type: Type.STRING } }
          },
          required: ["title", "techStack", "concept", "keyFeatures"]
        },
        roadmap30Days: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Immediate learning goals, tool setup and tutorials for the next 30 days"
        },
        roadmap60Days: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Project building and intermediate skill additions for the next 60 days"
        },
        roadmap90Days: {
          type: Type.ARRAY,
          items: { type: Type.STRING },
          description: "Certifications prep, portfolio polish, and job application tuning for the next 90 days"
        }
      },
      required: [
        "targetRoleAnalyzed", "overallGapPercentage", "missingSkills", 
        "missingToolsAndPlatforms", "missingCertifications", "recommendedProjectIdea",
        "roadmap30Days", "roadmap60Days", "roadmap90Days"
      ]
    };

    const promptText = `
Candidate Current Profile:
- Role Title: ${profile.title}
- Skills: ${profile.skills?.join(", ")}
- Email: ${profile.email}
- About: ${profile.about}
- Existing Certifications: ${profile.certifications?.join(", ")}
- Projects Highlights: ${JSON.stringify(profile.projects)}

Target Aspired Role: "${targetRole}"

Please run a comprehensive Gap Analysis. Identify specific differences in what the market and hiring managers expect for "${targetRole}" versus the candidate's current profile. Highlight missing skills (rated by critical, high, or beneficial importance), technical tools, recommended certifications with concrete providers, draft a powerful project idea that directly addresses the skill deficit, and design a fully structured 30-60-90 day upgrade roadmap.
Return valid JSON adhering to the target schema. No markdown formatting.`;

    const sysMsg = `You are the CareerSensei Resume vs Market Gap Centrifuge. You are precise, meticulous, and expert. You provide practical, technical, and concrete roadmap feedback for software and technical roles. Do not use vague advice. Give clear tools, tech stacks, and step-by-step goals.`;

    const result = await executeQueryPrompt(promptText, gapSchema, sysMsg);
    if (!result) throw new Error("Empty gap analysis from model.");
    res.json(JSON.parse(result));
  } catch (error: any) {
    console.error("Error in gap-analysis API:", error);
    res.status(500).json({ error: error.message || "Failed to perform market gap analysis." });
  }
});

// 3. Recommended Jobs Endpoint
app.post("/api/job-recommendations", async (req, res) => {
  try {
    const { profile, targetRole } = req.body;
    if (!profile) {
      return res.status(400).json({ error: "No candidate profile was supplied." });
    }

    const jobRecommendationsSchema = {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          id: { type: Type.STRING, description: "Unique short string ID" },
          title: { type: Type.STRING, description: "Job Title (e.g. Senior Frontend Architect)" },
          company: { type: Type.STRING, description: "Company Name (realistic or actual hiring partner)" },
          location: { type: Type.STRING, description: "Location (e.g. San Francisco, CA or Remote)" },
          matchScore: { type: Type.INTEGER, description: "A realistic match score based on current skills out of 100" },
          salaryRange: { type: Type.STRING, description: "Estimated salary (e.g. $120,000 - $155,000)" },
          whyYouFit: { type: Type.STRING, description: "Paragraph explaining why the candidate matches based on their achievements" },
          missingSkillsForThisRole: { type: Type.ARRAY, items: { type: Type.STRING }, description: "Specific missing tools or skills requested in typical job posts for this role" },
          atsOptimizationTip: { type: Type.STRING, description: "Tailored recommendation on how to reword their resume or highlight specific impacts for this role" },
          officialSource: { type: Type.STRING, description: "Realistic hub or origin career page (e.g., Stripe Careers, Vercel Careers, Remote.com)" }
        },
        required: [
          "id", "title", "company", "location", "matchScore", "salaryRange", 
          "whyYouFit", "missingSkillsForThisRole", "atsOptimizationTip", "officialSource"
        ]
      }
    };

    const promptText = `
Candidate Extracted Profile:
- Current Title: ${profile.title}
- Skills: ${profile.skills?.join(", ")}
- Projects Highlights: ${JSON.stringify(profile.projects)}
- Experience: ${JSON.stringify(profile.experience)}
- Target Role Interest: ${targetRole || profile.title}

Generate 4 highly relevant, highly personalized matching job listings in premier/trusted modern companies (e.g., tech giants, innovative startups, established mid-size enterprises) tailored to their target space. Ensure each contains a mathematically realistic Match Score (reflecting their skills vs role requirements) and genuine tailored advice on why they fit, what skills are missing for this specific role, and an actionable ATS alignment tip. Return strict valid JSON matching the schema.`;

    const sysMsg = `You are the CareerSensei Elite Matching Engine. You create realistic, highly motivating, and accurate job recommendations based on genuine developer qualities. No vague text. Return purely JSON arrays.`;

    const result = await executeQueryPrompt(promptText, jobRecommendationsSchema, sysMsg);
    if (!result) throw new Error("Empty job recommendation payload.");
    res.json(JSON.parse(result));
  } catch (error: any) {
    console.error("Error recommendations API:", error);
    res.status(500).json({ error: error.message || "Failed to fetch relative job opportunities." });
  }
});

// 4. Chat Career Mentor Coach Endpoint
app.post("/api/chat", async (req, res) => {
  try {
    const { messages, profile, targetRole } = req.body;
    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: "Invalid dynamic message history payload." });
    }

    const ai = getGeminiClient();
    const chat = ai.chats.create({
      model: "gemini-3.5-flash",
      config: {
        systemInstruction: `You are the CareerSensei Automated AI Mentor. You speak with specialized professional composure, quirky intelligence, and absolute honesty (no corporate fluff, pure architectural engineering truth). 
Candidate Details:
- Name: ${profile?.name || "The Candidate"}
- Profile Title: ${profile?.title || "Specialist"}
- Skills: ${profile?.skills?.join(", ") || "Technical generalist"}
- Target Role: ${targetRole || profile?.title || "Unspecified Target"}

Answer queries comprehensively, giving bold, practical tips on standard resume layouts, wording (e.g. passive vs action verbs), how to negotiate offers, prepare for tough coding interviews, or what side-projects to compile to stand out. Keep statements concise, beautifully structured with bullet points where appropriate, and highly actionable.`,
      },
    });

    // We can feed historic state context. Let's send the latest question or feed it sequentially.
    // The client sends the full chat array, we can use the last message or replay the conversation.
    const lastMsg = messages[messages.length - 1];
    
    // In @google/genai chat.sendMessage takes only the most recent message, but stores internal state if reused, 
    // or we can construct a unified conversation block to ensure stateless context is perfectly updated.
    let dialogueBlock = "";
    const subset = messages.slice(-8); // take last 8 exchanges to avoid overflowing context limits
    for (const m of subset) {
      dialogueBlock += `${m.role === "user" ? "User" : "Mentor"}: ${m.content}\n`;
    }
    dialogueBlock += `Mentor: (Generate direct answer now)`;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: dialogueBlock,
      config: {
        systemInstruction: `You are the CareerSensei AI Mentor. Keep answers warm, objective, professional, and dense with practical career blueprints.`
      }
    });

    res.json({ text: response.text });
  } catch (error: any) {
    console.error("Error in AI mentor chatbot chat:", error);
    res.status(500).json({ error: error.message || "Failed to process chat query." });
  }
});

// Health check route
app.get("/api/health", (req, res) => {
  res.json({ status: "healthy", timestamp: new Date().toISOString() });
});

// Vite server integrations
async function startServer() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
    console.log("Joined development mode. Vite middlewares registered.");
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
    console.log("Joined production mode. Serving static build from dist/.");
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server fully initialized and running at http://localhost:${PORT}`);
  });
}

startServer();
