import os, time, random
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import data_utils
import datasets
import eval_utils
from eczema_binary_classification_model import EczemaBinaryClassificationModel


seed = 13
EPSILON = 1e-8

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def log(s, filepath=None, display=True):
    '''
    Logs a message to console and optionally to a file

    Args:
        s : str
            message to log
        filepath : str or None
            if provided, append the message to this file
        display : bool
            if set, also print to console
    '''

    if display:
        print(s)

    if filepath is not None:
        with open(filepath, 'a+') as o:
            o.write(s + '\n')


def train(# Input file paths
          train_image_paths_file,
          train_labels_file,
          val_image_paths_file,
          val_labels_file,
          # Input settings
          batch_size=32,
          n_height=224,
          n_width=224,
          normalized_image_range=[0, 1],
          # Augmentation settings
          random_brightness=[-1, -1],
          random_contrast=[-1, -1],
          random_saturation=[-1, -1],
          random_hue=[-1, -1],
          random_flip_type=['none'],
          random_rotate_max=0,
          random_crop=False,
          # Encoder settings
          encoder_type='resnet50',
          pretrained=True,
          # Training settings
          balance_sampler=False,
          learning_rates=[1e-4],
          learning_schedule=[],
          learning_rate_scheduler='cosine_annealing',
          warmup_updates=0,
          warmup_init_lr=1e-7,
          adam_betas=[0.9, 0.999],
          weight_decay=1e-4,
          n_epoch=100,
          # Loss settings
          loss_func_name='cross_entropy',
          loss_class_weights=None,
          # Checkpoint and summary settings
          checkpoint_dirpath='checkpoints',
          n_step_per_checkpoint=1000,
          n_step_per_summary=100,
          n_display=5,
          start_step_validation=0,
          restore_path=None,
          # Hardware settings
          device='gpu',
          n_thread=8):
    '''
    Training loop for eczema binary classification

    Args:
        train_image_paths_file : str
            path to file containing training image paths
        train_labels_file : str
            path to file containing training labels
        val_image_paths_file : str
            path to file containing validation image paths
        val_labels_file : str
            path to file containing validation labels
        batch_size : int
            batch size for training
        n_height : int
            height to resize images to
        n_width : int
            width to resize images to
        normalized_image_range : list[float]
            [min, max] range for image normalization
        random_brightness : list[float]
            brightness augmentation range
        random_contrast : list[float]
            contrast augmentation range
        random_saturation : list[float]
            saturation augmentation range
        random_hue : list[float]
            hue augmentation range
        random_flip_type : list[str]
            flip augmentation types
        random_rotate_max : float
            max rotation angle for augmentation
        encoder_type : str
            encoder backbone architecture
        pretrained : bool
            use ImageNet pretrained weights
        learning_rates : list[float]
            learning rates for the schedule
        learning_schedule : list[int]
            epoch boundaries for learning rate changes
        learning_rate_scheduler : str
            scheduler type: cosine_annealing, multi_step
        warmup_updates : int
            number of warmup steps
        warmup_init_lr : float
            initial learning rate during warmup
        adam_betas : list[float]
            Adam optimizer beta parameters
        weight_decay : float
            weight decay for optimizer
        n_epoch : int
            number of training epochs
        loss_func_name : str
            loss function: cross_entropy, weighted_cross_entropy
        loss_class_weights : list[float] or None
            class weights for weighted cross entropy
        checkpoint_dirpath : str
            directory for saving checkpoints
        n_step_per_checkpoint : int
            steps between checkpoints
        n_step_per_summary : int
            steps between TensorBoard summaries
        n_display : int
            number of display samples per summary
        start_step_validation : int
            step to begin running validation
        restore_path : str or None
            path to restore model checkpoint from
        device : str
            device to use: gpu, cpu
        n_thread : int
            number of data loading threads
    '''

    '''
    Set up hardware and paths
    '''
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    # Set up checkpoint and event paths
    os.makedirs(checkpoint_dirpath, exist_ok=True)
    checkpoint_filepath = os.path.join(checkpoint_dirpath, 'model-{}.pth')
    best_checkpoint_filepath = os.path.join(checkpoint_dirpath, 'model-best.pth')
    log_path = os.path.join(checkpoint_dirpath, 'results.txt')
    event_path = os.path.join(checkpoint_dirpath, 'tensorboard')
    os.makedirs(event_path, exist_ok=True)

    '''
    Log settings
    '''
    log('Training settings:', log_path)
    log('  train_image_paths_file: {}'.format(train_image_paths_file), log_path)
    log('  train_labels_file:      {}'.format(train_labels_file), log_path)
    log('  val_image_paths_file:   {}'.format(val_image_paths_file), log_path)
    log('  val_labels_file:        {}'.format(val_labels_file), log_path)
    log('  batch_size:             {}'.format(batch_size), log_path)
    log('  n_height:               {}'.format(n_height), log_path)
    log('  n_width:                {}'.format(n_width), log_path)
    log('  normalized_image_range: {}'.format(normalized_image_range), log_path)
    log('  random_crop:            {}'.format(random_crop), log_path)
    log('  encoder_type:           {}'.format(encoder_type), log_path)
    log('  pretrained:             {}'.format(pretrained), log_path)
    log('  balance_sampler:        {}'.format(balance_sampler), log_path)
    log('  learning_rates:         {}'.format(learning_rates), log_path)
    log('  learning_schedule:      {}'.format(learning_schedule), log_path)
    log('  learning_rate_scheduler:{}'.format(learning_rate_scheduler), log_path)
    log('  warmup_updates:         {}'.format(warmup_updates), log_path)
    log('  warmup_init_lr:         {}'.format(warmup_init_lr), log_path)
    log('  adam_betas:             {}'.format(adam_betas), log_path)
    log('  weight_decay:           {}'.format(weight_decay), log_path)
    log('  n_epoch:                {}'.format(n_epoch), log_path)
    log('  loss_func:              {}'.format(loss_func_name), log_path)
    log('  loss_class_weights:     {}'.format(loss_class_weights), log_path)
    log('  checkpoint_dirpath:     {}'.format(checkpoint_dirpath), log_path)
    log('  n_step_per_checkpoint:  {}'.format(n_step_per_checkpoint), log_path)
    log('  n_step_per_summary:     {}'.format(n_step_per_summary), log_path)
    log('  start_step_validation:  {}'.format(start_step_validation), log_path)
    log('  device:                 {}'.format(device), log_path)
    log('  n_thread:               {}'.format(n_thread), log_path)
    log('', log_path)

    '''
    Set up datasets and dataloaders
    '''
    # Load and combine training images and labels from multiple filepaths
    if isinstance(train_image_paths_file, str):
        train_image_paths_file = [train_image_paths_file]
    train_image_paths = []
    for filepath in train_image_paths_file:
        train_image_paths.extend(data_utils.read_paths(filepath))

    if isinstance(train_labels_file, str):
        train_labels_file = [train_labels_file]
    train_labels = []
    for filepath in train_labels_file:
        train_labels.extend(data_utils.read_paths(filepath))

    # Load and combine validation images and labels from multiple filepaths
    if isinstance(val_image_paths_file, str):
        val_image_paths_file = [val_image_paths_file]
    val_image_paths = []
    for filepath in val_image_paths_file:
        val_image_paths.extend(data_utils.read_paths(filepath))

    if isinstance(val_labels_file, str):
        val_labels_file = [val_labels_file]
    val_labels = []
    for filepath in val_labels_file:
        val_labels.extend(data_utils.read_paths(filepath))

    train_dataset = datasets.EczemaDataset(
        dataset='train',
        image_paths=train_image_paths,
        labels=train_labels,
        shape=(n_height, n_width),
        random_crop=random_crop)

    val_dataset = datasets.EczemaDataset(
        dataset='val',
        image_paths=val_image_paths,
        labels=val_labels,
        shape=(n_height, n_width),
        random_crop=random_crop)

    n_train_sample = len(train_dataset)
    n_val_sample = len(val_dataset)

    n_train_step = n_epoch * int(np.ceil(n_train_sample / batch_size))

    log('Number of training samples:   {}'.format(n_train_sample), log_path)
    log('Number of validation samples: {}'.format(n_val_sample), log_path)
    log('Number of training steps:     {}'.format(n_train_step), log_path)
    log('', log_path)

    # Set up class balanced sampler if requested
    if balance_sampler:
        from collections import Counter
        from torch.utils.data import WeightedRandomSampler
        
        # Count classes in training dataset
        class_counts = Counter(train_dataset.labels)
        # Class weights: inverse of count
        class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
        # Compute weight for each sample
        sample_weights = [class_weights[lbl] for lbl in train_dataset.labels]
        sample_weights = torch.tensor(sample_weights, dtype=torch.float32)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True)
        
        train_shuffle = False
        log('Using WeightedRandomSampler for training loader imbalance balance.', log_path)
    else:
        train_sampler = None
        train_shuffle = True

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=n_thread,
        drop_last=True,
        pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        drop_last=False)

    '''
    Set up the model
    '''
    model = EczemaBinaryClassificationModel(
        encoder_type=encoder_type,
        pretrained=pretrained,
        normalized_image_range=normalized_image_range,
        device=device)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    log('Model parameters:             {}'.format(n_params), log_path)
    log('Trainable parameters:         {}'.format(n_trainable), log_path)
    log('', log_path)

    '''
    Set up optimizer and scheduler
    '''
    learning_rate = learning_rates[0]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=tuple(adam_betas),
        weight_decay=weight_decay)

    # Set up learning rate scheduler
    if learning_rate_scheduler == 'cosine_annealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=n_train_step - warmup_updates,
            eta_min=1e-7)
    elif learning_rate_scheduler == 'multi_step':
        # Convert epoch milestones to step milestones
        steps_per_epoch = int(np.ceil(n_train_sample / batch_size))
        milestones = [epoch * steps_per_epoch for epoch in learning_schedule]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=0.1)
    else:
        raise ValueError('Unsupported scheduler: {}'.format(learning_rate_scheduler))

    # Set up warmup scheduler
    if warmup_updates > 0:
        warmup_factor = warmup_init_lr / learning_rate

        def warmup_lambda(step):
            if step < warmup_updates:
                return warmup_factor + (1.0 - warmup_factor) * (step / warmup_updates)
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=warmup_lambda)
    else:
        warmup_scheduler = None

    # Restore model if path is provided
    start_step = 0

    if restore_path is not None and restore_path != '':
        start_step = model.restore_model(restore_path, optimizer)
        log('Restored model from: {}'.format(restore_path), log_path)
        log('Starting from step:  {}'.format(start_step), log_path)
        log('', log_path)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model.data_parallel()
        log('Using {} GPUs'.format(torch.cuda.device_count()), log_path)

    model.to(device)

    # Set up TensorBoard summary writers
    train_summary_writer = SummaryWriter(os.path.join(event_path, 'train'))
    val_summary_writer = SummaryWriter(os.path.join(event_path, 'val'))

    '''
    Set up best results tracking
    '''
    best_results = {
        'step': -1,
        'accuracy': -1.0,
        'precision': -1.0,
        'recall': -1.0,
        'f1': -1.0,
        'specificity': -1.0,
        'auc_roc': -1.0
    }

    '''
    Training loop
    '''
    train_step = start_step
    time_start = time.time()

    log('Begin training...', log_path)

    model.train()

    for epoch in range(1, n_epoch + 1):

        train_iter = tqdm(
            train_dataloader,
            total=len(train_dataloader),
            bar_format='{l_bar}{bar:10}{r_bar}',
            position=0,
            leave=True)

        for inputs in train_iter:

            # Move tensors to device
            inputs = [
                in_.to(device) if isinstance(in_, torch.Tensor) else in_
                for in_ in inputs
            ]

            image, label = inputs
            label = label.squeeze()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = model.forward(image)

            # Compute loss
            loss, loss_info = model.compute_loss(
                logits=logits,
                labels=label,
                loss_func=loss_func_name,
                class_weights=loss_class_weights)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Optimizer step
            optimizer.step()

            # Update learning rate
            if warmup_scheduler is not None and train_step < warmup_updates:
                warmup_scheduler.step()
            else:
                scheduler.step()

            train_step = train_step + 1

            # Update progress bar
            current_lr = optimizer.param_groups[0]['lr']
            train_iter.set_description(
                'Loss: {:.5f} / Epoch: {}/{} / Step: {}/{} / LR: {:.2e}'.format(
                    loss.item(),
                    epoch,
                    n_epoch,
                    train_step,
                    n_train_step,
                    current_lr))

            # Log training summary to TensorBoard
            if train_step > 0 and (train_step % n_step_per_summary) == 0:

                train_scalars = loss_info.copy()
                train_scalars['learning_rate'] = current_lr

                model.log_summary(
                    summary_writer=train_summary_writer,
                    tag='train',
                    step=train_step,
                    images=image.detach().clone(),
                    scalars=train_scalars,
                    n_image_per_summary=n_display)

            # Checkpoint and validation
            if train_step > 0 and (train_step % n_step_per_checkpoint) == 0:

                time_elapse = (time.time() - time_start) / 3600
                time_remain = (n_train_step - train_step + start_step) * time_elapse / max(train_step - start_step, 1)

                log('Step={:6}/{}  Loss={:.7f}  Time Elapsed={:.2f}h  Time Remaining={:.2f}h'.format(
                    train_step,
                    n_train_step,
                    loss.item(),
                    time_elapse,
                    time_remain),
                    log_path)

                # Run validation
                do_validation = train_step >= start_step_validation

                if do_validation:
                    model.eval()

                    with torch.no_grad():
                        best_results = validate(
                            model=model,
                            dataloader=val_dataloader,
                            step=train_step,
                            best_results=best_results,
                            loss_func_name=loss_func_name,
                            loss_class_weights=loss_class_weights,
                            device=device,
                            summary_writer=val_summary_writer,
                            n_display=n_display,
                            log_path=log_path)

                    # Save best model
                    if best_results['step'] == train_step:
                        model.save_model(best_checkpoint_filepath, train_step, optimizer)
                        log('Saved best model at step {}'.format(train_step), log_path)

                    model.train()

                # Save regular checkpoint
                model.save_model(
                    checkpoint_filepath.format(train_step),
                    train_step,
                    optimizer)

    '''
    Final validation and checkpoint
    '''
    log('Training complete.', log_path)

    time_elapse = (time.time() - time_start) / 3600
    log('Total training time: {:.2f}h'.format(time_elapse), log_path)

    # Final validation
    model.eval()

    with torch.no_grad():
        best_results = validate(
            model=model,
            dataloader=val_dataloader,
            step=train_step,
            best_results=best_results,
            loss_func_name=loss_func_name,
            loss_class_weights=loss_class_weights,
            device=device,
            summary_writer=val_summary_writer,
            n_display=n_display,
            log_path=log_path)

    # Save best model if final step is best
    if best_results['step'] == train_step:
        model.save_model(best_checkpoint_filepath, train_step, optimizer)

    # Save final checkpoint
    model.save_model(
        checkpoint_filepath.format(train_step),
        train_step,
        optimizer)

    # Log best results
    log('Best validation results:', log_path)
    log('  Step:        {}'.format(best_results['step']), log_path)
    log('  Accuracy:    {:.5f}'.format(best_results['accuracy']), log_path)
    log('  Precision:   {:.5f}'.format(best_results['precision']), log_path)
    log('  Recall:      {:.5f}'.format(best_results['recall']), log_path)
    log('  F1:          {:.5f}'.format(best_results['f1']), log_path)
    log('  Specificity: {:.5f}'.format(best_results['specificity']), log_path)
    log('  AUC-ROC:     {:.5f}'.format(best_results['auc_roc']), log_path)
    log('', log_path)

    # Close TensorBoard writers
    train_summary_writer.close()
    val_summary_writer.close()


def run(# Input file paths
        image_paths_file,
        labels_file,
        checkpoint_path,
        # Input settings
        batch_size=32,
        n_height=224,
        n_width=224,
        normalized_image_range=[0, 1],
        # Encoder settings
        encoder_type='resnet50',
        # Output settings
        output_path=None,
        save_outputs=False,
        # Hardware settings
        device='gpu',
        n_thread=8):
    '''
    Inference/evaluation loop for eczema binary classification

    Args:
        image_paths_file : str
            path to file containing image paths
        labels_file : str
            path to file containing labels
        checkpoint_path : str
            path to model checkpoint
        batch_size : int
            batch size for inference
        n_height : int
            image height
        n_width : int
            image width
        normalized_image_range : list[float]
            [min, max] range for image normalization
        encoder_type : str
            encoder backbone architecture
        output_path : str or None
            directory to save results
        save_outputs : bool
            whether to save prediction outputs
        device : str
            device to use: gpu, cpu
        n_thread : int
            number of data loading threads
    '''

    '''
    Set up hardware
    '''
    if device == 'cuda' or device == 'gpu':
        device = torch.device('cuda')
    elif device == 'cpu':
        device = torch.device('cpu')
    else:
        raise ValueError('Unsupported device: {}'.format(device))

    '''
    Set up output paths
    '''
    if output_path is not None:
        os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, 'results.txt') if output_path is not None else None

    log('Inference settings:', log_path)
    log('  image_paths_file:       {}'.format(image_paths_file), log_path)
    log('  labels_file:            {}'.format(labels_file), log_path)
    log('  checkpoint_path:        {}'.format(checkpoint_path), log_path)
    log('  batch_size:             {}'.format(batch_size), log_path)
    log('  n_height:               {}'.format(n_height), log_path)
    log('  n_width:                {}'.format(n_width), log_path)
    log('  encoder_type:           {}'.format(encoder_type), log_path)
    log('  output_path:            {}'.format(output_path), log_path)
    log('  device:                 {}'.format(device), log_path)
    log('', log_path)

    '''
    Set up dataset and dataloader
    '''
    # Load and combine test images and labels from multiple filepaths
    if isinstance(image_paths_file, str):
        image_paths_file = [image_paths_file]
    image_paths = []
    for filepath in image_paths_file:
        image_paths.extend(data_utils.read_paths(filepath))

    if isinstance(labels_file, str):
        labels_file = [labels_file]
    labels = []
    for filepath in labels_file:
        labels.extend(data_utils.read_paths(filepath))

    test_dataset = datasets.EczemaDataset(
        dataset='test',
        image_paths=image_paths,
        labels=labels,
        shape=(n_height, n_width))

    n_test_sample = len(test_dataset)

    log('Number of test samples: {}'.format(n_test_sample), log_path)
    log('', log_path)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_thread,
        drop_last=False)

    '''
    Set up and restore model
    '''
    model = EczemaBinaryClassificationModel(
        encoder_type=encoder_type,
        pretrained=False,
        normalized_image_range=normalized_image_range,
        device=device)

    model.restore_model(checkpoint_path)
    model.to(device)
    model.eval()

    log('Restored model from: {}'.format(checkpoint_path), log_path)
    log('', log_path)

    '''
    Run inference
    '''
    all_predictions = []
    all_probabilities = []
    all_labels = []

    log('Running inference...', log_path)

    test_iter = tqdm(
        test_dataloader,
        total=len(test_dataloader),
        bar_format='{l_bar}{bar:10}{r_bar}',
        position=0,
        leave=True)

    with torch.no_grad():
        for inputs in test_iter:

            # Move tensors to device
            inputs = [
                in_.to(device) if isinstance(in_, torch.Tensor) else in_
                for in_ in inputs
            ]

            image, label = inputs
            label = label.squeeze()

            # Forward pass
            logits = model.forward(image)

            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)

            all_predictions.append(predictions.cpu().numpy())
            all_probabilities.append(probabilities[:, 1].cpu().numpy())
            all_labels.append(label.cpu().numpy())

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    '''
    Evaluate results
    '''
    results = eval_utils.evaluate(
        predictions=all_predictions,
        probabilities=all_probabilities,
        labels=all_labels)

    log('Evaluation results:', log_path)
    log('  Accuracy:    {:.5f}'.format(results['accuracy']), log_path)
    log('  Precision:   {:.5f}'.format(results['precision_macro']), log_path)
    log('  Recall:      {:.5f}'.format(results['recall_macro']), log_path)
    log('  F1:          {:.5f}'.format(results['f1_macro']), log_path)
    log('  Specificity: {:.5f}'.format(results['specificity']), log_path)
    log('  AUC-ROC:     {:.5f}'.format(results['auc_roc']), log_path)
    log('  TP: {}  FP: {}  TN: {}  FN: {}'.format(
        results['tp'], results['fp'], results['tn'], results['fn']), log_path)
    log('', log_path)

    log('Confusion Matrix:', log_path)
    log('{}'.format(results['confusion_matrix']), log_path)
    log('', log_path)

    '''
    Save outputs
    '''
    if save_outputs and output_path is not None:
        predictions_filepath = os.path.join(output_path, 'predictions.txt')
        probabilities_filepath = os.path.join(output_path, 'probabilities.txt')

        # Save predictions
        with open(predictions_filepath, 'w') as f:
            for idx in range(len(all_predictions)):
                f.write('{} {} {}\n'.format(
                    image_paths[idx],
                    int(all_predictions[idx]),
                    float(all_probabilities[idx])))

        # Save probabilities
        with open(probabilities_filepath, 'w') as f:
            for idx in range(len(all_probabilities)):
                f.write('{}\n'.format(float(all_probabilities[idx])))

        log('Saved predictions to: {}'.format(predictions_filepath), log_path)
        log('Saved probabilities to: {}'.format(probabilities_filepath), log_path)

    log('Inference complete.', log_path)


def validate(model,
             dataloader,
             step,
             best_results,
             loss_func_name,
             loss_class_weights,
             device,
             summary_writer,
             n_display,
             log_path):
    '''
    Runs validation, computes metrics, logs results, and tracks best model

    Args:
        model : EczemaBinaryClassificationModel
            the model to evaluate
        dataloader : DataLoader
            validation dataloader
        step : int
            current training step
        best_results : dict
            dictionary tracking best validation results
        loss_func_name : str
            loss function name (cross_entropy or weighted_cross_entropy)
        loss_class_weights : list[float] or None
            class weights for weighted cross entropy
        device : torch.device
            device to run inference on
        summary_writer : SummaryWriter or None
            TensorBoard writer for validation
        n_display : int
            number of sample images to log
        log_path : str
            path to results log file
    Returns:
        dict : updated best_results
    '''

    all_predictions = []
    all_probabilities = []
    all_labels = []
    val_losses = []

    for inputs in dataloader:
        # Move to device
        inputs = [
            in_.to(device) if isinstance(in_, torch.Tensor) else in_
            for in_ in inputs
        ]

        image, label = inputs
        label = label.squeeze()

        # Forward pass
        logits = model.forward(image)

        # Compute loss
        loss, _ = model.compute_loss(
            logits=logits,
            labels=label,
            loss_func=loss_func_name,
            class_weights=loss_class_weights)

        val_losses.append(loss.item())

        # Get predictions and probabilities
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities[:, 1].detach().cpu().numpy())
        all_labels.append(label.cpu().numpy())

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_probabilities = np.concatenate(all_probabilities, axis=0)
    all_labels = np.concatenate(all_labels, axis=0).astype(np.int64)

    mean_val_loss = np.mean(val_losses)

    # Compute all metrics
    results = eval_utils.evaluate(
        predictions=all_predictions,
        probabilities=all_probabilities,
        labels=all_labels)

    # Log results
    log('Validation results at step {}:'.format(step), log_path)
    log('  Loss:        {:.5f}'.format(mean_val_loss), log_path)
    log('  Accuracy:    {:.5f}'.format(results['accuracy']), log_path)
    log('  Precision:   {:.5f}'.format(results['precision_macro']), log_path)
    log('  Recall:      {:.5f}'.format(results['recall_macro']), log_path)
    log('  F1:          {:.5f}'.format(results['f1_macro']), log_path)
    log('  Specificity: {:.5f}'.format(results['specificity']), log_path)
    log('  AUC-ROC:     {:.5f}'.format(results['auc_roc']), log_path)
    log('  TP: {}  FP: {}  TN: {}  FN: {}'.format(
        results['tp'], results['fp'], results['tn'], results['fn']), log_path)
    log('', log_path)

    # Log to TensorBoard
    if summary_writer is not None:
        val_scalars = {
            'loss': mean_val_loss,
            'accuracy': results['accuracy'],
            'precision_macro': results['precision_macro'],
            'recall_macro': results['recall_macro'],
            'f1_macro': results['f1_macro'],
            'specificity': results['specificity'],
            'auc_roc': results['auc_roc']
        }

        model.log_summary(
            summary_writer=summary_writer,
            tag='val',
            step=step,
            scalars=val_scalars,
            n_image_per_summary=n_display)

    # Track best results based on improvement across multiple metrics (F1, AUC-ROC, etc.)
    n_improve = 0
    if results['accuracy'] > best_results['accuracy']:
        n_improve += 1
    if results['precision_macro'] > best_results['precision']:
        n_improve += 1
    if results['recall_macro'] > best_results['recall']:
        n_improve += 1
    if results['f1_macro'] > best_results['f1']:
        n_improve += 1
    if results['specificity'] > best_results['specificity']:
        n_improve += 1
    if results['auc_roc'] > best_results['auc_roc']:
        n_improve += 1

    # Update if we have at least 2 improved metrics, or if it is the first validation step
    if n_improve >= 2 or best_results['step'] == -1:
        best_results['accuracy'] = results['accuracy']
        best_results['precision'] = results['precision_macro']
        best_results['recall'] = results['recall_macro']
        best_results['f1'] = results['f1_macro']
        best_results['specificity'] = results['specificity']
        best_results['auc_roc'] = results['auc_roc']
        best_results['step'] = step

        log('New best results at step {} (improved {} metrics):'.format(step, n_improve), log_path)
        log('  Accuracy:    {:.5f}'.format(best_results['accuracy']), log_path)
        log('  Precision:   {:.5f}'.format(best_results['precision']), log_path)
        log('  Recall:      {:.5f}'.format(best_results['recall']), log_path)
        log('  F1:          {:.5f}'.format(best_results['f1']), log_path)
        log('  Specificity: {:.5f}'.format(best_results['specificity']), log_path)
        log('  AUC-ROC:     {:.5f}'.format(best_results['auc_roc']), log_path)
        log('', log_path)

    return best_results