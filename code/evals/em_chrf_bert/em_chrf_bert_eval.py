import pandas as pd
import json
from io import StringIO
import numpy as np
import bert_score
from sacrebleu import sentence_chrf
from utils.eval import *
from utils.utils import *
from utils.table_utils import *
import torch
import traceback 
import re
import argparse
import os


def parse_table(text):
    tables = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Look for a table header line (e.g., "### Team" or "### Player")
        header_match = re.match(r"^###\s+(.+)", line)
        if header_match:
            table_name = header_match.group(1).strip()
            table_lines = []
            i += 1
            # Skip any blank lines immediately following the table header.
            while i < len(lines) and not lines[i].strip():
                i += 1
            # Collect lines that start with a pipe
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1

            if table_lines:
                # Remove any separator rows (rows that are made mostly of dashes, colons, or pipes)
                content_lines = [ln for ln in table_lines if not re.match(r"^\s*\|[\s\-\|:]+\|\s*$", ln)]
                if not content_lines:
                    continue

                # Process the header row: empty cells become None.
                header_line = content_lines[0]
                headers = [
                    cell.strip() if cell.strip() != "" else None
                    for cell in header_line.strip().strip('|').split('|')
                ]
                
                data = []
                # Process the remaining lines as data rows.
                for row_line in content_lines[1:]:
                    row_cells = [
                        cell.strip() if cell.strip() != "" else None
                        for cell in row_line.strip().strip('|').split('|')
                    ]
                    # Ensure that each row has the same number of cells as headers.
                    if len(row_cells) < len(headers):
                        row_cells.extend([None] * (len(headers) - len(row_cells)))
                    elif len(row_cells) > len(headers):
                        row_cells = row_cells[:len(headers)]
                    data.append(row_cells)
                
                if data:
                    df = pd.DataFrame(data, columns=headers, dtype=object)
                    df.replace('None', np.nan, inplace=True)
                    df = df[~df.iloc[:, 1:].isna().all(axis=1)]
                    df.dropna(axis=1, how='all', inplace=True)
                    tables[table_name] = df
        else:
            i += 1

    return tables


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path', 
        type=str,
        required=True,
        help='Input tables path'
    )
    parser.add_argument(
        '--output_path', 
        type=str,
        required=True,
        help='Output evals path'
    )
    parser.add_argument(
        '--gold_path',
        type=str,
        required=True,
        help='Gold table path'
    )
    args = parser.parse_args()

    method_path = args.input_path
    eval_path = args.output_path

    gold_player_path = os.path.join(args.gold_path, 'player/')
    gold_team_path = os.path.join(args.gold_path, 'team/')
    eval_dict = {}

    error_idx = []
    for res_key in range(728):
        try:
            print(res_key)
            res_key = str(res_key)
            if os.path.getsize(os.path.join(gold_player_path, f'{res_key}.csv')) == 0:
                gold_player = pd.DataFrame()
            else:
                gold_player = pd.read_csv(gold_player_path + f'{int(res_key)}.csv', keep_default_na=False, na_values=[''],engine='python', dtype=object)
            
            
            if os.path.getsize(os.path.join(gold_team_path, f'{res_key}.csv')) == 0:
                gold_team = pd.DataFrame()
            else:
                gold_team = pd.read_csv(gold_team_path + f'{int(res_key)}.csv', keep_default_na=False, na_values=[''],engine='python', dtype=object)
            
            
            if gold_team.empty:
                gold_team = np.array([[]])
            else:
                gold_team_arr = [gold_team.columns.tolist()] + gold_team.values.tolist()
                gold_team_arr[0][0] = ''
                gold_team = np.array(gold_team_arr)
            
            if gold_player.empty:
                gold_player = np.array([[]])
            else:
                gold_player_arr = [gold_player.columns.tolist()] + gold_player.values.tolist()
                gold_player_arr[0][0] = ''
                gold_player = np.array(gold_player_arr)
            
            with open(os.path.join(method_path, f'{res_key}.txt'), 'r') as f:
                text = f.read().strip()
            tables = parse_table(text)
            team, player = tables.get('Team Table'), tables.get('Player Table')


            if team is None:
                team = np.array([[]])
            elif isinstance(team, pd.DataFrame) and team.empty:
                team = np.array([[]])
            else:
                team_arr = [team.columns.tolist()] + team.values.tolist()
                team_arr[0][0] = ''
                team = np.array(team_arr)

            if player is None:
                player = np.array([[]])
            elif isinstance(player, pd.DataFrame) and player.empty:
                player = np.array([[]])
            else:
                player_arr = [player.columns.tolist()] + player.values.tolist()
                player_arr[0][0] = ''
                player = np.array(player_arr)
            

            metrics_team = evaluate_table(model_output=team, gold_label=gold_team, eval_bert=True, eval_chrf=True, eval_exact_match=True)
            metrics_players = evaluate_table(model_output=player, gold_label=gold_player, eval_bert=True, eval_chrf=True, eval_exact_match=True)

            eval_dict['Test_pt' + res_key] = {}
                    
            eval_dict['Test_pt' + res_key]['Team'] = metrics_team
            eval_dict['Test_pt' + res_key]['Player'] = metrics_players

        except Exception as e:
            print('Idx: ', res_key)
            print(e)
            error_idx.append(int(res_key))

    with open(eval_path, 'w') as f:
        json.dump(eval_dict, f, indent=6)
        
    print(error_idx)
    print(len(error_idx))