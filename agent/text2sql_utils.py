import pandas as pd
import numpy as np
import os

from llm.llm.chatgpt import ChatGPT, OpenAIWrapper
from llm.llm.gemini import Gemini

from llm.llm_utils import get_code_from_text_response
from pydantic import BaseModel, ConfigDict
from typing import Union
import re


class Table(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    table: Union[pd.DataFrame, str, None]
    sql: str = ""
    description: str = ""
    
    def __str__(self):
        return f"Table(desc = {self.description})"
    
    def __repr__(self):
        return f"Table(desc = {self.description})"
    

def table_to_markdown(table: Table|pd.DataFrame|str, max_string = 5000) -> str:
    
    if table is None:
        return ""
    
    # If it's a string
    if isinstance(table, str):
        return table

    if not isinstance(table, list):
        table = [table]
        
    markdown = ""
    for t in table:
        if isinstance(table, pd.DataFrame):
            markdown += df_to_markdown(t)[:max_string] + "\n\n"
        
        try:
            markdown += f"**{t.description}**\n\n"
            markdown += df_to_markdown(t.table)[:max_string] + "\n\n"
        
        except:
            raise ValueError("Invalid table type")
    
    return markdown
    
    
def join_and_get_difference(df1, df2):
    
    main_cols = ''
    for col in df1.columns:
        if col in ['category_code', 'universal_code', 'stock_code', 'ratio_code']:
            main_cols = col
            break
    if main_cols == '':
        return df1, df2
    
    # If find main column
    
    diff = df2[~df2[main_cols].isin(df1[main_cols])]
    df1 = pd.concat([df1, diff])
    return df1, diff


def get_llm_wrapper(model_name, **kwargs):
    if 'gpt' in model_name:
        return ChatGPT(model_name=model_name, **kwargs)
    
    elif 'gemini' in model_name:
        return Gemini(model_name=model_name, **kwargs)
    
    return OpenAIWrapper(model_name=model_name, **kwargs)
    


def read_file_without_comments(file_path, start=["#", "//"]):
    if not os.path.exists(file_path):
        Warning(f"File {file_path} not found")
        return ""
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            if not any([line.startswith(s) for s in start]):
                new_lines.append(line)
        return ''.join(new_lines)
    
def read_file(file_path):
    if not os.path.exists(file_path):
        Warning(f"File {file_path} not found")
        return ""
    
    with open(file_path, 'r') as f:
        return f.read()
    
    
def df_to_markdown(df):
    if not isinstance(df, pd.DataFrame):
        return str(df)
    markdown = df.to_markdown(index=False)
    return markdown


def company_name_to_stock_code(db, names, method = 'similarity', top_k = 2) -> pd.DataFrame:
    """
    Get the stock code based on the company name
    """
    if not isinstance(names, list):
        names = [names]
    
    if method == 'similarity': # Using similarity search
        df = db.return_company_info(names, top_k)
        df.drop_duplicates(subset=['stock_code'], inplace=True)
        return df
    
    else: # Using rational DB
        dfs = []
        query = "SELECT * FROM company WHERE company_name LIKE '%{name}%'"
        
        if method == 'bm25-ts':
            query = "SELECT stock_code, company_name FROM company_info WHERE to_tsvector('english', company_name) @@ to_tsquery('{name}');"
        
        elif 'bm25' in method:
            pass # Using paradeDB
        
        else:
            raise ValueError("Method not supported")  
        
        for name in names:
            
            # Require translate company name in Vietnamese to English
            name = name # translate(name, 'vi', 'en')
            query = query.format(name=name)
            result = db.query(query, return_type='dataframe')
            
            dfs.append(result)
            
        if len(dfs) > 0:
            result = pd.concat(dfs)
        else:
            result = pd.DataFrame(columns=['stock_code', 'company_name'])
        return result
    
    
def is_sql_full_of_comments(sql_text):
    lines = sql_text.strip().splitlines()
    comment_lines = 0
    total_lines = len(lines)
    in_multiline_comment = False

    for line in lines:
        stripped_line = line.strip()
        
        # Check if it's a single-line comment or empty line
        if stripped_line.startswith('--') or not stripped_line:
            comment_lines += 1
            continue
        
        # Check for multi-line comments
        if stripped_line.startswith('/*'):
            in_multiline_comment = True
            comment_lines += 1
            # If it ends on the same line
            if stripped_line.endswith('*/'):
                in_multiline_comment = False
            continue
        
        if in_multiline_comment:
            comment_lines += 1
            if stripped_line.endswith('*/'):
                in_multiline_comment = False
            continue

    # Check if comment lines are the majority of lines
    return comment_lines >= total_lines  
    
    
def get_table_name_from_sql(sql_text):
    pattern = r"-- ###\s*(.+)"
    matches = re.findall(pattern, sql_text)
    if len(matches) > 0:
        return matches[0]
    return ""
    
    
def TIR_reasoning(response, db, verbose=False):
    codes = get_code_from_text_response(response)
        
    TIR_response = ""
    execution_error = []
    execution_table = []
    
    sql_code = []
    
    for code in codes:
        if code['language'] == 'sql':
            codes = code['code'].split(";")
            for content in codes:
                # clean the content
                if content.strip() != "":
                    sql_code.append(content)
            
    for i, code in enumerate(sql_code):    
        if verbose:    
            print(f"SQL Code {i+1}: \n{code}")
        
        if not is_sql_full_of_comments(code): 
            name = get_table_name_from_sql(code)
              
            table = db.query(code, return_type='dataframe')
            
            # If it see an error in the SQL code
            if isinstance(table, str):
                execution_error.append((i, table))
                continue
            
            table_obj = Table(table=table, sql=code, description=f"SQL Result {i+1}: {name}")
            
            execution_table.append(table_obj)
    
    
    return execution_error, execution_table

    
def get_company_detail_from_df(dfs, db, method = 'similarity') -> pd.DataFrame:
    stock_code = set()
    if not isinstance(dfs, list):
        dfs = [dfs]
    
    if isinstance(dfs[0], Table):
        dfs = [df.table for df in dfs]
    
    for df in dfs:
        for col in df.columns:
            if col == 'stock_code':
                stock_code.update(df[col].tolist())
            if col == 'company_name':
                stock_code.update(company_name_to_stock_code(db, df[col].tolist(), method)['stock_code'].tolist())
            if col == 'invest_on':
                stock_code.update(company_name_to_stock_code(db, df[col].tolist(), method)['stock_code'].tolist())
            
    list_stock_code = list(stock_code)
    
    return company_name_to_stock_code(db, list_stock_code, method)
    
 
def check_openai_response(messages):
    if len(messages) == 0:
        return False
    
    for message in messages:
        if message.get('role', '') not in ['assistant', 'system', 'user']:
            return False
    
    return True
    
    
def reformat_messages(messages):
    
    if not check_openai_response(messages):
        raise ValueError("Invalid messages")
    
    flag_system = False
    system_message = ""
    if messages[0].get('role','') == 'system':
        flag_system = True
        system_message = messages[0]['content']
        messages = messages[1:]
        
    new_messages = []
    for i, message in enumerate(messages):
        role = message.get('role', 'user')
        content = message.get('content', '')
        
        if i == 0 and flag_system:
            content =  f"<<SYS>>\n\n{system_message}\n\n<<SYS>>\n\n" + content
        
        new_messages.append({
            'role': role,
            'content': content
        })
    if new_messages[-1].get('role', '') == 'user':
        new_messages.append(
            {
                'role': 'assistant',
                'content': "Something went wrong, please try again"
            }
        )
        
    return new_messages
            
            
def _prune_entity(table: pd.DataFrame, entities: list[str]):
    entities = np.array(entities)
    if not isinstance(entities, list):
        entities = [entities]

    cols = table.columns
    table['mask'] = 0
    
    for col in cols:
        print(col)
        if col in ['is_bank','is_securities'] and col in table.columns:
            table.drop(col, axis=1, inplace=True)
            continue
            
        mask_contain_entities = np.isin(table[col].values, entities)
        table.loc[mask_contain_entities, 'mask'] += 1
        
    table = table[table['mask'] > 0]
    table = table.drop('mask', axis=1)
    
    return table
            
            
def prune_unnecessary_data_from_sql(tables: list[Table], messages: list[dict]): 
    is_list = True
    if not isinstance(tables, list):
        is_list = False
        tables = [tables]
        
    assistant_messages = []
    for message in messages:
        if message.get('role', '') == 'assistant':
            assistant_messages.append(message.get('content', ''))
            
    sql_codes = []
    for message in assistant_messages:
        codes = get_code_from_text_response(message)
        for code in codes:
            if code['language'] == 'sql':
                sql_codes.append(code['code'])
    
    mentioned_entities = set()
    
    for code in sql_codes:
        matches = re.findall(r"'(.*?)'", code)
        mentioned_entities.update(matches)
    
    for table in tables:
        table.table = _prune_entity(table.table.copy(), list(mentioned_entities))
        
    if not is_list:
        return tables[0]
    return tables
    
def check_null_table(tables: Table|pd.DataFrame):
    
    if isinstance(tables, pd.DataFrame):
        if tables is None or tables.empty:
            return True
        return False
    
    if isinstance(tables, Table):
        if tables.table is None or tables.table.empty:
            return True
        return False
    
    
def prune_null_table(tables: list[Table|pd.DataFrame]):
    new_tables = []
    for table in tables:
        
        if check_null_table(table):
            continue
        new_tables.append(table)
        
    return new_tables