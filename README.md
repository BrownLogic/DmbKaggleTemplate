# DmbKaggleTemplate
This is a rough framework for organizing models and submissions.  Still a work in process.

Steps for use:
1.  Make a copy of the DmbKaggleTemplate folder and place in target location.
2.  Rename the DmbKaggleTemplate to new name ('BNP_Claims_Competition') for example  
3.  Delete ./.git folder
4.  Delete ./.idea folder
5.  Replace any files in the ./data folder with the project files provided.
6.  Delete all .pynb files and .ipynb checkpoint folders.  
7.  Delete all files in ./Run_001/saved_objects and in ./Run_001/submissions
8.  Empty contents of ./Run_001/execution_and_error.log
9.  Update the ./Run_001/README.md file
10.  Review and update the contents of ./SETTINGS.json file.  You may need to update this further after you've analysed the data.  For example, you will need to identify the numeric, string and non feature columns.  The directories *should* remain the same.
11.  Open PyCharm and go to File-Open.  Go to the new folder you created and open it in a new window or to current one.
12.  In PyCharm, goto VCS-Import Into Version Control-Create Git Repository.  Create the repository in the newly created folder.  (alternately, you could do this from the git shell (
    git init
    git add .gitignore
    git add README.md
    git commit -m 'initial project version'
13.  In GitHub, create a new repository (existing folder)
14.  In git shell (under the appropriate directory) type:
    git remote add origin {URL of new repository}
    git push -u origin master

15.  In Pycharm, select the new project.  Go to VCS-Commit Changes.  Select the appropriate files and directories and Commit-Push.


