# COMP5530M-Group-Project-Inflation-Forecasting

Repository for group 12's 2024/25 academic year COMP5530M group project on inflation forecasting with machine learning models.

## Contents
- Scoping document (deadline: semester 1 end)
  - Associated documentation (e.g. reading list)
- Project code
- Project documentation
- Final deliverables (deadline: 09/05/2025)
  - Report (with project management appendix)
  - Showcase video
  - Showcase poster (deadline: poster session)

## Execution Instructions
1. Ensure current working directory matches the directory of this file: 'pwd' or equivalent to establish location, then 'cd' or equivalent to this directory
2. (optional) Create a virtual environment: `python -m venv ./'
3. Ensure relevant dependencies are installed: `pip install -r requirements.txt'

    i. If working in a virtual environment, ensure this command is being run in that context (terminals often prefix the current working directory with (.venv) to show this)
4. Browse the notebooks and run as you wish, ensuring the kernel used matches that with which you installed the dependencies


## Git rules:
- Make pull request in git and get someone to review changes.

- Important to do this every time!
  - Switch to branch: git checkout "\<branchname>"
  - Pull from master to get changes: git merge origin main
  - Push changes: git push

- Creating branches:
  - Create new branch: git checkout -b "\<branchname>"
  - Set upstream: git push --set-upstream origin "\<branchname>"

- Adding files and commiting
  - Add files: git add .
  - Commit messages: git commit -m "message"
  - Push changes: git push
  
- Swithing to someone elses branch for PR
  - git switch "\<branchname>"

## File Naming Conventions:  (can be changed)
- "{ModelName}_{info}.{pth|pkl}"
- Make sure weights file is the same name as the model in the "Weights" folder
