{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "3c8389fb",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:00.743514Z",
     "iopub.status.busy": "2024-10-05T01:20:00.742922Z",
     "iopub.status.idle": "2024-10-05T01:20:01.757201Z",
     "shell.execute_reply": "2024-10-05T01:20:01.756247Z"
    },
    "papermill": {
     "duration": 1.026001,
     "end_time": "2024-10-05T01:20:01.759922",
     "exception": false,
     "start_time": "2024-10-05T01:20:00.733921",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "import numpy as np \n",
    "import pandas as pd \n",
    "\n",
    "# import os\n",
    "# for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "#     for filename in filenames:\n",
    "#         print(os.path.join(dirname, filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "506ee6b8",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:01.773546Z",
     "iopub.status.busy": "2024-10-05T01:20:01.772625Z",
     "iopub.status.idle": "2024-10-05T01:20:05.754932Z",
     "shell.execute_reply": "2024-10-05T01:20:05.753899Z"
    },
    "papermill": {
     "duration": 3.991667,
     "end_time": "2024-10-05T01:20:05.757543",
     "exception": false,
     "start_time": "2024-10-05T01:20:01.765876",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df1 = pd.read_csv('/kaggle/input/luflow-network-intrusion-detection-data-set/2022/06/2022.06.12/2022.06.12.csv')\n",
    "df2 = pd.read_csv('/kaggle/input/luflow-network-intrusion-detection-data-set/2022/06/2022.06.13/2022.06.13.csv')\n",
    "df3 = pd.read_csv('/kaggle/input/luflow-network-intrusion-detection-data-set/2022/06/2022.06.14/2022.06.14.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "f8f6dc1b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:05.771960Z",
     "iopub.status.busy": "2024-10-05T01:20:05.771551Z",
     "iopub.status.idle": "2024-10-05T01:20:05.815808Z",
     "shell.execute_reply": "2024-10-05T01:20:05.814762Z"
    },
    "papermill": {
     "duration": 0.054252,
     "end_time": "2024-10-05T01:20:05.818286",
     "exception": false,
     "start_time": "2024-10-05T01:20:05.764034",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "16"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_dataset = pd.concat([df1, df2, df3])\n",
    "df_dataset.reset_index(drop=True, inplace=True)\n",
    "len(df_dataset.columns)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "b33df7da",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:05.832350Z",
     "iopub.status.busy": "2024-10-05T01:20:05.831650Z",
     "iopub.status.idle": "2024-10-05T01:20:06.000516Z",
     "shell.execute_reply": "2024-10-05T01:20:05.999249Z"
    },
    "papermill": {
     "duration": 0.17923,
     "end_time": "2024-10-05T01:20:06.003569",
     "exception": false,
     "start_time": "2024-10-05T01:20:05.824339",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1068376 entries, 0 to 1068375\n",
      "Data columns (total 16 columns):\n",
      " #   Column         Non-Null Count    Dtype  \n",
      "---  ------         --------------    -----  \n",
      " 0   avg_ipt        1068376 non-null  float64\n",
      " 1   bytes_in       1068376 non-null  int64  \n",
      " 2   bytes_out      1068376 non-null  int64  \n",
      " 3   dest_ip        1068376 non-null  int64  \n",
      " 4   dest_port      964168 non-null   float64\n",
      " 5   entropy        1068376 non-null  float64\n",
      " 6   num_pkts_out   1068376 non-null  int64  \n",
      " 7   num_pkts_in    1068376 non-null  int64  \n",
      " 8   proto          1068376 non-null  int64  \n",
      " 9   src_ip         1068376 non-null  int64  \n",
      " 10  src_port       964168 non-null   float64\n",
      " 11  time_end       1068376 non-null  int64  \n",
      " 12  time_start     1068376 non-null  int64  \n",
      " 13  total_entropy  1068376 non-null  float64\n",
      " 14  label          1068376 non-null  object \n",
      " 15  duration       1068376 non-null  float64\n",
      "dtypes: float64(6), int64(9), object(1)\n",
      "memory usage: 130.4+ MB\n"
     ]
    }
   ],
   "source": [
    "df_dataset.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "22b5413a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:06.017986Z",
     "iopub.status.busy": "2024-10-05T01:20:06.017559Z",
     "iopub.status.idle": "2024-10-05T01:20:06.100404Z",
     "shell.execute_reply": "2024-10-05T01:20:06.099451Z"
    },
    "papermill": {
     "duration": 0.093182,
     "end_time": "2024-10-05T01:20:06.103224",
     "exception": false,
     "start_time": "2024-10-05T01:20:06.010042",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_dataset = df_dataset.drop(['num_pkts_out', 'num_pkts_in'], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "8119b01a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:06.117950Z",
     "iopub.status.busy": "2024-10-05T01:20:06.117526Z",
     "iopub.status.idle": "2024-10-05T01:20:06.124663Z",
     "shell.execute_reply": "2024-10-05T01:20:06.123649Z"
    },
    "papermill": {
     "duration": 0.017728,
     "end_time": "2024-10-05T01:20:06.127349",
     "exception": false,
     "start_time": "2024-10-05T01:20:06.109621",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['avg_ipt', 'bytes_in', 'bytes_out', 'dest_ip', 'dest_port', 'entropy',\n",
       "       'proto', 'src_ip', 'src_port', 'time_end', 'time_start',\n",
       "       'total_entropy', 'label', 'duration'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_dataset.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "8b0b4718",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:06.142794Z",
     "iopub.status.busy": "2024-10-05T01:20:06.142361Z",
     "iopub.status.idle": "2024-10-05T01:20:06.523720Z",
     "shell.execute_reply": "2024-10-05T01:20:06.522436Z"
    },
    "papermill": {
     "duration": 0.392365,
     "end_time": "2024-10-05T01:20:06.526582",
     "exception": false,
     "start_time": "2024-10-05T01:20:06.134217",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "8d5a1147",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:06.541926Z",
     "iopub.status.busy": "2024-10-05T01:20:06.541481Z",
     "iopub.status.idle": "2024-10-05T01:20:06.737398Z",
     "shell.execute_reply": "2024-10-05T01:20:06.736203Z"
    },
    "papermill": {
     "duration": 0.206816,
     "end_time": "2024-10-05T01:20:06.740204",
     "exception": false,
     "start_time": "2024-10-05T01:20:06.533388",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_dataset.dropna(inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "4ba82763",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:06.756936Z",
     "iopub.status.busy": "2024-10-05T01:20:06.755963Z",
     "iopub.status.idle": "2024-10-05T01:20:07.605375Z",
     "shell.execute_reply": "2024-10-05T01:20:07.604117Z"
    },
    "papermill": {
     "duration": 0.859834,
     "end_time": "2024-10-05T01:20:07.607829",
     "exception": false,
     "start_time": "2024-10-05T01:20:06.747995",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4447\n"
     ]
    }
   ],
   "source": [
    "print(df_dataset.duplicated().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "0e5ae519",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:07.623506Z",
     "iopub.status.busy": "2024-10-05T01:20:07.623046Z",
     "iopub.status.idle": "2024-10-05T01:20:08.424233Z",
     "shell.execute_reply": "2024-10-05T01:20:08.423036Z"
    },
    "papermill": {
     "duration": 0.812361,
     "end_time": "2024-10-05T01:20:08.427135",
     "exception": false,
     "start_time": "2024-10-05T01:20:07.614774",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_dataset.drop_duplicates(inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "1b78cbd9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:08.443087Z",
     "iopub.status.busy": "2024-10-05T01:20:08.442645Z",
     "iopub.status.idle": "2024-10-05T01:20:09.206706Z",
     "shell.execute_reply": "2024-10-05T01:20:09.205508Z"
    },
    "papermill": {
     "duration": 0.775189,
     "end_time": "2024-10-05T01:20:09.209480",
     "exception": false,
     "start_time": "2024-10-05T01:20:08.434291",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "print(df_dataset.duplicated().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "7dd4624b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:09.225814Z",
     "iopub.status.busy": "2024-10-05T01:20:09.224975Z",
     "iopub.status.idle": "2024-10-05T01:20:10.014978Z",
     "shell.execute_reply": "2024-10-05T01:20:10.013941Z"
    },
    "papermill": {
     "duration": 0.801101,
     "end_time": "2024-10-05T01:20:10.017675",
     "exception": false,
     "start_time": "2024-10-05T01:20:09.216574",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "df_dataset.drop_duplicates(inplace = True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "23c0e737",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:10.034033Z",
     "iopub.status.busy": "2024-10-05T01:20:10.033616Z",
     "iopub.status.idle": "2024-10-05T01:20:10.804019Z",
     "shell.execute_reply": "2024-10-05T01:20:10.802942Z"
    },
    "papermill": {
     "duration": 0.781632,
     "end_time": "2024-10-05T01:20:10.806783",
     "exception": false,
     "start_time": "2024-10-05T01:20:10.025151",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0\n"
     ]
    }
   ],
   "source": [
    "print(df_dataset.duplicated().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "10e8d036",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:10.824447Z",
     "iopub.status.busy": "2024-10-05T01:20:10.823468Z",
     "iopub.status.idle": "2024-10-05T01:20:11.007679Z",
     "shell.execute_reply": "2024-10-05T01:20:11.006456Z"
    },
    "papermill": {
     "duration": 0.195535,
     "end_time": "2024-10-05T01:20:11.010009",
     "exception": false,
     "start_time": "2024-10-05T01:20:10.814474",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "label\n",
       "benign       516220\n",
       "outlier      365385\n",
       "malicious     78116\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 14,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_dataset[\"label\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "a2f8c1fe",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:11.027395Z",
     "iopub.status.busy": "2024-10-05T01:20:11.026939Z",
     "iopub.status.idle": "2024-10-05T01:20:11.206947Z",
     "shell.execute_reply": "2024-10-05T01:20:11.205951Z"
    },
    "papermill": {
     "duration": 0.19194,
     "end_time": "2024-10-05T01:20:11.209627",
     "exception": false,
     "start_time": "2024-10-05T01:20:11.017687",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "old_value = 'outlier'\n",
    "new_value = 0\n",
    "df_dataset['label'] = df_dataset['label'].replace(old_value, new_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "7cc0cb57",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:11.227951Z",
     "iopub.status.busy": "2024-10-05T01:20:11.226599Z",
     "iopub.status.idle": "2024-10-05T01:20:11.401258Z",
     "shell.execute_reply": "2024-10-05T01:20:11.400297Z"
    },
    "papermill": {
     "duration": 0.186621,
     "end_time": "2024-10-05T01:20:11.404020",
     "exception": false,
     "start_time": "2024-10-05T01:20:11.217399",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "old_value = 'malicious'\n",
    "new_value = 1\n",
    "df_dataset['label'] = df_dataset['label'].replace(old_value, new_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "0827ca8b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:11.421561Z",
     "iopub.status.busy": "2024-10-05T01:20:11.420764Z",
     "iopub.status.idle": "2024-10-05T01:20:12.027366Z",
     "shell.execute_reply": "2024-10-05T01:20:12.026216Z"
    },
    "papermill": {
     "duration": 0.618315,
     "end_time": "2024-10-05T01:20:12.030135",
     "exception": false,
     "start_time": "2024-10-05T01:20:11.411820",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_17/3502624556.py:3: FutureWarning: Downcasting behavior in `replace` is deprecated and will be removed in a future version. To retain the old behavior, explicitly call `result.infer_objects(copy=False)`. To opt-in to the future behavior, set `pd.set_option('future.no_silent_downcasting', True)`\n",
      "  df_dataset['label'] = df_dataset['label'].replace(old_value, new_value)\n"
     ]
    }
   ],
   "source": [
    "old_value = 'benign'\n",
    "new_value = 2\n",
    "df_dataset['label'] = df_dataset['label'].replace(old_value, new_value)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "b49b59e9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:12.047904Z",
     "iopub.status.busy": "2024-10-05T01:20:12.047470Z",
     "iopub.status.idle": "2024-10-05T01:20:12.067258Z",
     "shell.execute_reply": "2024-10-05T01:20:12.066151Z"
    },
    "papermill": {
     "duration": 0.031848,
     "end_time": "2024-10-05T01:20:12.070086",
     "exception": false,
     "start_time": "2024-10-05T01:20:12.038238",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "label\n",
       "2    516220\n",
       "0    365385\n",
       "1     78116\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df_dataset[\"label\"].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "753f7a5e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2024-10-05T01:20:12.089089Z",
     "iopub.status.busy": "2024-10-05T01:20:12.087982Z",
     "iopub.status.idle": "2024-10-05T01:22:36.806413Z",
     "shell.execute_reply": "2024-10-05T01:22:36.804874Z"
    },
    "papermill": {
     "duration": 144.731335,
     "end_time": "2024-10-05T01:22:36.809691",
     "exception": false,
     "start_time": "2024-10-05T01:20:12.078356",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/tmp/ipykernel_17/323831337.py:19: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  y_train = torch.tensor(y_train, dtype=torch.long)\n",
      "/tmp/ipykernel_17/323831337.py:20: UserWarning: To copy construct from a tensor, it is recommended to use sourceTensor.clone().detach() or sourceTensor.clone().detach().requires_grad_(True), rather than torch.tensor(sourceTensor).\n",
      "  y_test = torch.tensor(y_test, dtype=torch.long)\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch [2/100], Loss: 1.0198\n",
      "Epoch [4/100], Loss: 0.9586\n",
      "Epoch [6/100], Loss: 0.9031\n",
      "Epoch [8/100], Loss: 0.8522\n",
      "Epoch [10/100], Loss: 0.8059\n",
      "Epoch [12/100], Loss: 0.7653\n",
      "Epoch [14/100], Loss: 0.7315\n",
      "Epoch [16/100], Loss: 0.7048\n",
      "Epoch [18/100], Loss: 0.6834\n",
      "Epoch [20/100], Loss: 0.6646\n",
      "Epoch [22/100], Loss: 0.6479\n",
      "Epoch [24/100], Loss: 0.6347\n",
      "Epoch [26/100], Loss: 0.6247\n",
      "Epoch [28/100], Loss: 0.6159\n",
      "Epoch [30/100], Loss: 0.6082\n",
      "Epoch [32/100], Loss: 0.6018\n",
      "Epoch [34/100], Loss: 0.5954\n",
      "Epoch [36/100], Loss: 0.5895\n",
      "Epoch [38/100], Loss: 0.5849\n",
      "Epoch [40/100], Loss: 0.5809\n",
      "Epoch [42/100], Loss: 0.5779\n",
      "Epoch [44/100], Loss: 0.5756\n",
      "Epoch [46/100], Loss: 0.5732\n",
      "Epoch [48/100], Loss: 0.5713\n",
      "Epoch [50/100], Loss: 0.5698\n",
      "Epoch [52/100], Loss: 0.5685\n",
      "Epoch [54/100], Loss: 0.5674\n",
      "Epoch [56/100], Loss: 0.5665\n",
      "Epoch [58/100], Loss: 0.5658\n",
      "Epoch [60/100], Loss: 0.5650\n",
      "Epoch [62/100], Loss: 0.5642\n",
      "Epoch [64/100], Loss: 0.5636\n",
      "Epoch [66/100], Loss: 0.5630\n",
      "Epoch [68/100], Loss: 0.5625\n",
      "Epoch [70/100], Loss: 0.5621\n",
      "Epoch [72/100], Loss: 0.5617\n",
      "Epoch [74/100], Loss: 0.5613\n",
      "Epoch [76/100], Loss: 0.5610\n",
      "Epoch [78/100], Loss: 0.5607\n",
      "Epoch [80/100], Loss: 0.5604\n",
      "Epoch [82/100], Loss: 0.5602\n",
      "Epoch [84/100], Loss: 0.5599\n",
      "Epoch [86/100], Loss: 0.5597\n",
      "Epoch [88/100], Loss: 0.5595\n",
      "Epoch [90/100], Loss: 0.5592\n",
      "Epoch [92/100], Loss: 0.5590\n",
      "Epoch [94/100], Loss: 0.5588\n",
      "Epoch [96/100], Loss: 0.5586\n",
      "Epoch [98/100], Loss: 0.5584\n",
      "Epoch [100/100], Loss: 0.5581\n",
      "Accuracy on test set: 75.85%\n"
     ]
    }
   ],
   "source": [
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "\n",
    "X = df_dataset.drop(['label', 'dest_ip', 'src_ip', 'dest_port', 'src_port', 'time_start', 'time_end'], axis=1).values\n",
    "y = df_dataset['label'].values\n",
    "\n",
    "scaler = StandardScaler()\n",
    "X = scaler.fit_transform(X)\n",
    "\n",
    "y = torch.tensor(y, dtype=torch.long)\n",
    "\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "X_train = torch.tensor(X_train, dtype=torch.float32)\n",
    "X_test = torch.tensor(X_test, dtype=torch.float32)\n",
    "y_train = torch.tensor(y_train, dtype=torch.long)\n",
    "y_test = torch.tensor(y_test, dtype=torch.long)\n",
    "\n",
    "class SimpleLinearModel(nn.Module):\n",
    "    def __init__(self, input_size, output_size):\n",
    "        super(SimpleLinearModel, self).__init__()\n",
    "        self.linear1 = nn.Linear(input_size, 128)\n",
    "        self.linear2 = nn.Linear(128, 64)\n",
    "        self.linear3 = nn.Linear(64, 32)\n",
    "        self.linear4 = nn.Linear(32, output_size)\n",
    "    \n",
    "    def forward(self, x):\n",
    "        x = self.linear1(x)\n",
    "        x = self.linear2(x)\n",
    "        x = self.linear3(x)\n",
    "        x = self.linear4(x)\n",
    "        return x\n",
    "\n",
    "input_size = X_train.shape[1] \n",
    "output_size = 3 \n",
    "learning_rate = 0.001\n",
    "epochs = 100\n",
    "\n",
    "model = SimpleLinearModel(input_size, output_size)\n",
    "criterion = nn.CrossEntropyLoss()\n",
    "optimizer = optim.Adam(model.parameters(), lr=learning_rate)\n",
    "\n",
    "for epoch in range(epochs):\n",
    "    outputs = model(X_train)\n",
    "    loss = criterion(outputs, y_train)\n",
    "    \n",
    "    optimizer.zero_grad()\n",
    "    loss.backward()\n",
    "    optimizer.step()\n",
    "    \n",
    "    if (epoch+1) % 2 == 0:\n",
    "        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')\n",
    "\n",
    "with torch.no_grad():\n",
    "    outputs = model(X_test)\n",
    "    _, predicted = torch.max(outputs, 1)\n",
    "    accuracy = (predicted == y_test).sum().item() / len(y_test)\n",
    "    print(f'Accuracy on test set: {accuracy * 100:.2f}%')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0ebb6d43",
   "metadata": {
    "papermill": {
     "duration": 0.013266,
     "end_time": "2024-10-05T01:22:36.835977",
     "exception": false,
     "start_time": "2024-10-05T01:22:36.822711",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "datasetId": 975848,
     "sourceId": 9522107,
     "sourceType": "datasetVersion"
    }
   ],
   "dockerImageVersionId": 30786,
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.14"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 160.499315,
   "end_time": "2024-10-05T01:22:38.072871",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2024-10-05T01:19:57.573556",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
