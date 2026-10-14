# 🚀 Getting Started with LLM Engineering

## Welcome!

You've just completed the initial setup of the **LLM Engineer's Handbook** project. This guide will help you start your learning journey.

---

## ✅ Setup Complete!

Your environment is ready:
- ✅ Python 3.11.14 installed
- ✅ Virtual environment created (`.venv`)
- ✅ All dependencies installed (172+ packages)
- ✅ ZenML server running on port 8237
- ✅ MongoDB connected (localhost:27017)
- ✅ Qdrant connected (localhost:6333)

---

## 📚 Your Learning Journey Starts Here

### Step 1: Open the Documentation

The complete documentation is in the `docs/` folder:

```
docs/
├── README.md                      # Start here!
├── CURRICULUM.md                  # Chapter-by-chapter learning path
├── BOOK-MAP.md                    # Chapter -> session -> code map
├── GETTING_STARTED.md             # This file
├── CODEBASE-INTELLIGENCE.md       # Graphs, architecture, tooling
└── sessions/
    ├── session_1.1_project_overview.md
    ├── session_1.2_domain_layer.md
    ├── session_1.3_infrastructure_layer.md
    ├── session_2.1_web_crawling.md
    ├── session_2.2_text_preprocessing.md
    ├── session_2.3_feature_engineering.md
    ├── session_3.1_instruction_dataset.md
    ├── session_3.2_preference_dataset.md
    ├── session_4.1_advanced_rag.md
    ├── session_4.2_embedding_models.md
    ├── session_4.3_streaming_cdc.md
    ├── session_5.1_sft.md
    ├── session_5.2_dpo.md
    ├── session_5.3_sagemaker_deployment.md
    ├── session_5.4_data_curation.md
    ├── session_6.1_fastapi_api.md
    ├── session_6.2_rag_inference_flow.md
    ├── session_7.1_comet_ml.md
    ├── session_7.2_opik_monitoring.md
    ├── session_7.3_model_evaluation.md
    ├── session_7.4_rag_evaluation.md
    ├── session_8.1_docker.md
    ├── session_8.2_cicd.md
    ├── session_8.3_zenml.md
    ├── session_8.4_inference_optimization.md
    ├── session_9.1_data_warehouse.md
    ├── session_9.2_performance.md
    ├── session_9.3_security.md
    ├── session_10.1_deployment_topologies.md
    ├── session_11.1_ct_pipeline_alerting.md
    └── appendix_mlops_principles.md
```

### Step 2: Begin with Session 1.1

**Open**: [`docs/sessions/session_1.1_project_overview.md`](./sessions/session_1.1_project_overview.md)

This session covers:
- Project architecture overview
- Domain-Driven Design (DDD) principles
- How to navigate the codebase
- Environment setup verification

**Time**: 2-3 hours

---

## 🎯 Recommended Learning Path

### Week 1-2: Foundations

```bash
# Read Session 1.1
# Then read Session 1.2 (Domain Layer)
# Then read Session 1.3 (Infrastructure)

# Test your setup
.venv\Scripts\python -c "from llm_engineering.domain.base import NoSQLBaseDocument; print('✓ Setup OK')"
```

### Week 3-5: Data Engineering

```bash
# Read Session 2.1 (Web Crawling)
# Create your first crawler
# Run the ETL pipeline
.venv\Scripts\python -m tools.run --run-etl --no-cache
```

### Week 6-7: Dataset Generation

```bash
# Read Session 3.1 (Instruction Datasets)
# Generate your first dataset
.venv\Scripts\python -m tools.run --run-generate-instruct-datasets --no-cache
```

### Week 8-9: Advanced RAG

```bash
# Read Session 4.1 (Advanced RAG)
# Test the RAG system
.venv\Scripts\python -m tools.rag
```

---

## 🛠️ Quick Commands Reference

### Start Services

```bash
# ZenML Server (if not running)
zenml up --port 8237

# Check ZenML status
.venv\Scripts\zenml status

# View ZenML dashboard
# Open: http://localhost:8237
```

### Run Pipelines

```bash
# List available pipelines
.venv\Scripts\python -m tools.run --help

# Run ETL pipeline
.venv\Scripts\python -m tools.run --run-etl --no-cache

# Run feature engineering
.venv\Scripts\python -m tools.run --run-feature-engineering --no-cache

# Run RAG inference
.venv\Scripts\python -m tools.rag
```

### Access Dashboards

| Service | URL | Purpose |
|---------|-----|---------|
| **ZenML** | http://localhost:8237 | Pipeline monitoring |
| **Qdrant** | http://localhost:6333/dashboard | Vector database |
| **MongoDB** | Use MongoDB Compass | Document database |

---

## 📖 How to Use the Documentation

### For Beginners

1. **Start with CURRICULUM.md**: Understand the full learning path
2. **Follow Sessions Sequentially**: Each session builds on previous knowledge
3. **Code Along**: Don't just read—write the code yourself
4. **Complete Exercises**: Each session has hands-on tasks
5. **Build Projects**: Apply learning to capstone projects

### For Experienced Developers

1. **Browse README.md**: Find specific topics quickly
2. **Jump to Relevant Sessions**: Focus on what you need
3. **Study Code Examples**: Learn from real implementations
4. **Contribute**: Improve the documentation

---

## 🎓 Session Structure

Each session follows this format:

```markdown
# Session Title

## Learning Objectives
What you'll learn

## Architecture Overview
Visual diagrams and explanations

## Key Files Explained
Deep dive into code with examples

## Hands-On Exercises
Step-by-step coding tasks

## Knowledge Check
Quiz questions to test understanding

## Next Steps
Where to go next
```

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `zenml login` fails with authorization error

**Solution**: Don't use `zenml login <URL>` for local server. Just run:
```bash
zenml stack set default
```

---

**Issue**: Import errors with transformers

**Solution**: Fix dependency versions:
```bash
uv pip install "tokenizers>=0.22.0,<=0.23.0" --python 3.11
uv pip install "huggingface-hub>=0.34.0,<1.0" --python 3.11
```

---

**Issue**: MongoDB connection fails

**Solution**: Start Docker containers:
```bash
docker compose up -d
```

---

## 📞 Getting Help

### Documentation

- **README.md**: Main index and quick reference
- **CURRICULUM.md**: Complete learning path
- **Session Files**: Detailed tutorials

### Community

- **GitHub Issues**: Report bugs
- **Discord**: Join LLM engineering communities
- **Stack Overflow**: Ask technical questions

### Book Resources

- **LLM Engineer's Handbook**: [Amazon Link](https://www.amazon.com/LLM-Engineers-Handbook-engineering-production/dp/1836200072/)
- **Official Repository**: [GitHub](https://github.com/PacktPublishing/LLM-Engineers-Handbook)

---

## 🎯 Your First Task

**Right Now**: Open [`docs/sessions/session_1.1_project_overview.md`](./sessions/session_1.1_project_overview.md) and:

1. Read the "Architecture Overview" section
2. Study the "Project Structure Deep Dive"
3. Complete the "Hands-On: Environment Setup"
4. Answer the "Knowledge Check" questions

**Time**: 2-3 hours

**Outcome**: You'll understand the entire project structure

---

## 📈 Track Your Progress

Create a learning journal:

```markdown
# My Learning Journey

## Week 1
- [x] Session 1.1: Project Overview
- [ ] Session 1.2: Domain Layer
- [ ] Session 1.3: Infrastructure

## Notes
- Learned about DDD principles
- Understood dependency flow
- Set up environment successfully
```

---

## 🚀 Next Steps

After completing Session 1.1:

1. **Session 1.2**: Domain Layer - Data Modeling
2. **Session 1.3**: Infrastructure Layer - Database Connections
3. **Session 2.1**: Web Crawling with Selenium

Then choose your path:
- **Data Engineering Track**: Sessions 2.1-2.3
- **RAG Track**: Sessions 4.1-4.2
- **LLM Training Track**: Sessions 5.1-5.3

---

## 💡 Tips for Success

1. **Code Daily**: Even 30 minutes helps
2. **Build Projects**: Apply what you learn
3. **Teach Others**: Write blog posts or explain to friends
4. **Join Communities**: Connect with other learners
5. **Don't Rush**: Understanding takes time

---

## 🎉 You're Ready!

Everything is set up and ready to go. Your learning journey in LLM engineering starts now!

**Next Step**: Open [`docs/sessions/session_1.1_project_overview.md`](./sessions/session_1.1_project_overview.md) and begin!

---

**Questions?** Check the [README.md](./README.md) for more details.

**Need Help?** See the [CURRICULUM.md](./CURRICULUM.md) for the complete learning path.

Happy Learning! 🚀
