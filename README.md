# Run DSim
1. download the dataset, we upload the one dataset we use (ednet with 500 samples) in https://www.dropbox.com/scl/fo/72604nn56kkhahtl3jny4/AOL6tHJo02ZdRy3k2FiOZYU?rlkey=k577qlem5s6jpz8qgk3gkl7gu&st=pqiq9mzb&dl=0.
2. move the data you download to data/ednet500/
3. run the code with: python main.py


# Here are some issues you might encounter, you can address that with the solutions:
1. FileExistsError: [Errno 17] File exists: 'run_result/'
   
   please delete file run_result
   
2. FileNotFoundError: [Errno 2] No such file or directory: 'data/ednet500/problem_skill_maxSkillOfProblem_number.pkl'

   please check the path of dataset, ensure the path of dataset is correct
   
4. AssertionError: Torch not compiled with CUDA enabled
   ensure your cuda is available
