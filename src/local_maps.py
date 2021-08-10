## Contains: replacement dictionaries and useful lists used in the data processing. 

# dictionary with shorter strings for education levels
new_EdLevel = {'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)': 'Master’s degree',
 'Bachelor’s degree (B.A., B.S., B.Eng., etc.)': 'Bachelor’s degree',
 'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 'Secondary school',
 'Professional degree (JD, MD, etc.)': 'Professional degree',
 'Some college/university study without earning a degree': 'College study/no degree',
 'Associate degree (A.A., A.S., etc.)' : 'Associate degree',
 'Other doctoral degree (Ph.D., Ed.D., etc.)': 'Other doctoral degree',
 'I never completed any formal education' : 'No formal education'}
 
# dictionary with shorter descriptions for the undegraduate majors
new_UndergradMajor = {'Computer science, computer engineering, or software engineering':
                           'Computer science',
       'Another engineering discipline (such as civil, electrical, mechanical, etc.)':'Engineering other',
       'A humanities discipline (such as literature, history, philosophy, etc.)': 'Humanities',
       'A health science (such as nursing, pharmacy, radiology, etc.)': 'Health science',
       'Information systems, information technology, or system administration' : 'Information system',
       'Web development or web design': 'Web dev/design',
        'Mathematics or statistics': 'Math or stats',
       'A natural science (such as biology, chemistry, physics, etc.)': 'Natural science',
       'Fine arts or performing arts (such as graphic design, music, studio art, etc.)': 'Arts',
       'I never declared a major': 'No major',
       'A social science (such as anthropology, psychology, political science, etc.)': 'Social science',
       'A business discipline (such as accounting, finance, marketing, etc.)': 'Business'}
       
# replace some strings in the EdImpt column
new_EdImpt = {'Not at all important/not necessary': 'Not important'}

# encoding map for job satisfaction
JobSat_dict =  {'Very dissatisfied': 1, 'Slightly dissatisfied': 2,
               'Neither satisfied nor dissatisfied': 3, 
               'Slightly satisfied': 4, 'Very satisfied': 5}

# list of columns to be removed
cols_del = [
    # personal, demographics  information
    #'Respondent', 
    'MainBranch', 'Employment', 'Hobbyist', 
    'Country',
    'Ethnicity', 'Gender', 'Sexuality', 'Trans', 'Age',                                
    
    # related to ConvertedComp
    'CompFreq', 'CompTotal', 'CurrencyDesc', 'CurrencySymbol',                 
    
    # questions regarding future activities
    'DatabaseDesireNextYear', 'MiscTechDesireNextYear',                    
    'CollabToolsDesireNextYear', 'PlatformDesireNextYear',
    'LanguageDesireNextYear', 'WebframeDesireNextYear',
    
    # questions regarding this survey
    'SurveyEase', 'SurveyLength', 'WelcomeChange',                           
    
    # question regarding participation is StackOverflow
    'SOSites', 'SOComm', 'SOPartFreq',
    'SOVisitFreq', 'SOAccount',                                               

    # columns related to other columns
    'Age1stCode', 'YearsCodePro', 'DevClass', 
    
    # high cardinality, multiple choices columns, add noise 
    'DatabaseWorkedWith','MiscTechWorkedWith','LanguageWorkedWith',
    'WebframeWorkedWith', #'CollabToolsWorkedWith',                                                 

    # other questions not directly related to our goal
    'JobHunt', 
    'JobHuntResearch', 'Stuck',
    'PurchaseResearch', 
     #'PurchaseWhat', 
    'Stuck', 'PurpleLink',
    'OffTopic', 'OtherComms',
    'JobFactors', #'JobSeek',
    'DevType']                                                            


# the columns grouped by types in the predictors matrix

# numerical columns
num_cols = ['ConvertedComp', 'WorkWeekHrs', 'YearsCode']
# the list of discrete columns with many levels 
multi_cols = ['PlatformWorkedWith', 'CollabToolsWorkedWith']
# the list of discrete columns with several levels
uni_cols = ['EdLevel', 'EdImpt', 'OnboardGood', 'JobSeek', 
            'Overtime', 'DevOps', 'Learn', 'UndergradMajor', 'OpSys', 
            'DevOpsImpt', 'OrgSize', 'PurchaseWhat']

# the list of performance metrics associated to confusion matrix
metrics_list = ['accuracy','precision','recall', 'f1']

# the columns to keep from expansion after applying MultiLabelBinarizer
all_keep = ['Linux', 'Windows', 'Docker', 'Github', 'Slack', 'Jira']

