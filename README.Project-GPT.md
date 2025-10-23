# Overview of the GPT-2 ECE408 Final Project 

## GPT-2

[GPT-2](https://github.com/openai/gpt-2) (Generative Pretrained Transformer 2) is a transformer-based language model developed by OpenAI, released in 2019. GPT-2 is based on transformers, which was first introduced in the paper [Attention is All You Need](https://arxiv.org/pdf/1706.03762). It is one of the first large-scale models to showcase the power of unsupervised learning for natural language generation tasks. GPT-2 is also part of a lineage of models that leverage transformer architecture, focusing on autoregressive generation, meaning it predicts the next word in a sequence given all the previous words using a decoder-only architecture. This paradigm are proved to be scalable and adopted in later widely used works including chatGPT and others. Therefore, hardware optimization and acceleration based on GPT-2 or other decoder-only transformers has become an crucial topic in parallel programming application. 

Because of the significance of GPT-2 and its relatively manageable model size, you are tasked with using CUDA to accelerate GPT-2's inference (forward pass) performance.

## Project Details

This project is designed to be a team-based, multi-milestone project spanning the remainder of the semester. You need to form a team of 3-4 students (enrolled in ECE 408) to work on this project. Each team will be assigned a course staff member as their mentor/grader for the project. We recommend having a team of 4 students if possible, as this project is quite large in scope, and the workload for each student is significant. 

In this Project, you and your team will implement the entire forward pass of GPT-2 using CUDA, perform various optimizations, and thoroughly analyze the performance of your implementation using Nvidia profiling tools. In the last milestone you will also have the chance to explore a few additional optimizations of your choice and further improve the performance of your implementation. Throughout the project, your team's deliverables will include code submissions, code development logs/notes, written reports, milestone demo meetings, and a final presentation.

### Tentative Timeline and Grading Policy

| Milestone   | Tentative Deadline          |
| ----------- | ----------------------      |
| Milestone 1 | Week 7-8                    |
| Milestone 2 | Week 10-11                  |
| Milestone 3 | Week 15-16                  |

**Note**: The Grace Period Policy for assignment submissions applies to the GPT Final Project as well.

## Notes on Teamwork

In this project, you and your teammates will work, present, and be graded as a team. This means that all members share responsibility for the final outcome and are expected to contribute meaningfully to the project. It is essential that you always actively communicate with each other about your individual progress, challenges, and plans about the project. This is especially important given the required milestone demo meetings, as every group member can affect your grade in these meeting sessions.
