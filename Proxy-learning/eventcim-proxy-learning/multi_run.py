import torch
import torch.nn as nn
from multiprocess import run, reading_sequence, train_xytp, train_frame, model, get_number_of_correct_samples
from multiprocessing import Pool
from tqdm import tqdm
import time
from typing import List

# Define the number of process going to be used for parallelize
num_of_workers = 2


# Process pool
class Multi_process_batching:

    @staticmethod
    def generate_batch_based_reading_sequence(reading_sequences: List[int], batch: int) -> List[List[int]]:
        """
        reading_sequences: input list of reading indexes
        batch: batch size

        Return: lists of minibatch sample reading indexes
        """
        new_batch_sequence = []
        assert len(reading_sequences) // batch >= 1
        mini_batch_index = []
        for indexing in range(len(reading_sequences) // batch):
            if len(mini_batch_index) < batch:
                mini_batch_index.append(reading_sequences[indexing])
            else:
                new_batch_sequence.append(mini_batch_index)
                mini_batch_index = [reading_sequences[indexing]]
        return new_batch_sequence



# pbr = tqdm(reading_sequence)
correct_ann = 0
correct_simulator = 0
number_of_samples = 0

count = 0

multi_process_reading_index = Multi_process_batching.generate_batch_based_reading_sequence(reading_sequences=reading_sequence, batch=2)

optimizer = torch.optim.SGD(model.ann.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

pbr = tqdm(multi_process_reading_index)

correct_ann = 0
correct_simulator = 0
number_of_samples = 0
model.train()
print("process start")
tstart = time.time()
for multisample in multi_process_reading_index:

    # coverting ann state_dict to simulator
    model._update_snn()
    print(f"update time: {time.time() - tstart}")
    # multiprocessing of simulator
    snn_output = []

    for s in multisample:
        with Pool(num_of_workers) as p:
            xytp, label = train_xytp[s]
            ans = p.apply_async(run, args=(xytp,))  ###########
            snn_output.append(ans.get())
    p.close()
    p.join()
    print(f"simulator time: {time.time() - tstart}")
    # ann concatennate as batch
    frame, frame_label = train_frame[multisample[0]]
    frame_2, frame_label_2 = train_frame[multisample[1]]
    target = torch.tensor([frame_label, frame_label_2]).view(-1)
    target = target.to(model.device)
    tensor = torch.cat(
        [torch.tensor(frame).unsqueeze(0), torch.tensor(frame_2).unsqueeze(0)], 0
    )

    ann_output = model.frame_torch_forward(tensor)
    ann_activation = sum(model.ann_activation_record) / 2
    snn_tensor_output = torch.cat(
        [snn_output[0][0].view(1, -1), snn_output[1][0].view(1, -1)], 0
    )

    correct_ann += get_number_of_correct_samples(ann_output, target)
    correct_simulator += get_number_of_correct_samples(
        snn_tensor_output, target
    )
    print(f"ann forward time: {time.time() - tstart}")
    snn_activation = torch.tensor(0.5 * (sum(snn_output[0][1]) + sum(snn_output[1][1])))
    # empty the hook list
    model.ann_activation_record = []
    ann_output.data.copy_(snn_tensor_output)
    ann_activation.data.copy_(snn_activation)

    target_activation = 2e6
    proxy_activation_loss = (
            10
            * (1 / ann_activation ** 2)
            * torch.sqrt((ann_activation - target_activation) ** 2)
    ).to(model.device)
    if target_activation > ann_activation:
        loss = criterion(ann_output, target)
    else:
        loss = criterion(ann_output, target) + proxy_activation_loss

    optimizer.zero_grad()
    loss.backward()

    print(f"backward time: {time.time() - tstart}")

    number_of_samples += 2
    # pbr.set_description(
    #     f"train_ann:{round(correct_ann / number_of_samples, 4)}, "
    #     f"train_sim{round(correct_simulator / number_of_samples, 4)}",
    # )
