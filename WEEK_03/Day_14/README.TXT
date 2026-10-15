[ User Request ] 
       │
       ▼
┌──────────────┐      PRD Document
│  Product     ├──────────────────────────┐
│  Manager     │                          │
└──────────────┘                          ▼
                                 ┌─────────────────┐
                                 │    Architect    │
                                 └────────┬────────┘
                                          │
                               System Architecture & Specs
                                          │
                                          ▼
                                 ┌─────────────────┐
                                 │    Engineer     │◄────────────┐
                                 └────────┬────────┘             │
                                          │                      │
                                   Executable Code               │ Code Revision
                                          │                      │ Request
                                          ▼                      │
                                 ┌─────────────────┐             │
                                 │    QA / Test    ├─────────────┘
                                 │    Engineer     │ (If tests fail)
                                 └────────┬────────┘
                                          │
                                   Passed Tests
                                          ▼
                               ┌─────────────────────┐
                               │ Final Code / Output │
                               └─────────────────────┘