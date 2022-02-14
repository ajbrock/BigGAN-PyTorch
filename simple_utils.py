import torch
from torchvision import transforms



def load_checkpoint(model, filepath):

    checkpoint = torch.load(filepath)
    state_dict = checkpoint['state_dict']

    model.load_state_dict(state_dict)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()
    return model

def data_transform(train_folder, val_folder, test_folder, is_gist=False, is_saliency=False):
    # VGG-16 Takes 224x224 images as input, so we resize all of them

    image_size = 224
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_transforms = {
        train_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)],p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        val_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.RandomApply([transforms.RandomRotation(20)],p=.5),
            transforms.RandomApply([transforms.ColorJitter(hue=.05, saturation=.05)], p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]),
        test_folder: transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])
    }

    if is_gist:
        # this transform is not changing the image self so it can be used in test as well.
        data_transforms['gist'] = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            normalize,
            transforms.ToPILImage()
        ])
    if is_saliency:
        # this transform is not changing the image self so it can be used in test as well.
        data_transforms['saliency'] = transforms.Compose([
            transforms.Resize(size=(image_size, image_size)),
            transforms.ToTensor(),
            normalize,
            transforms.ToPILImage()
        ])
    return data_transforms

def reg_dataloader(data_dir, train_folder, val_folder, test_folder, csv_train, csv_val, csv_test,
                   batch_size, istrain, is_gist=False, is_saliency=False):
    """
           Args:
              istrain: is training or not
           """

    data_transforms = data_transform(train_folder, val_folder, test_folder, is_gist, is_saliency)

    csv_files = data_csv(train_folder, val_folder, test_folder, csv_train, csv_val, csv_test)

    transform_gist = None if not is_gist else data_transforms['gist']
    transform_saliency = None if not is_saliency else data_transforms['saliency']

    image_datasets = {
        x: RegressionDataset(
            csv_file=csv_files[x],
            root_dir=os.path.join(data_dir, x),
            transform=data_transforms[x],
            istrain=istrain,
            is_gist=is_gist,
            transform_gist=transform_gist,
            is_saliency=is_saliency,
            transform_saliency=transform_saliency
        )
        for x in ([train_folder, val_folder] if istrain else [test_folder])
    }

    if istrain:
        dataloaders = {
            x: DataLoader(
                image_datasets[x],
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True,
                sampler=ImbalancedDatasetSampler(image_datasets[x], callback_get_label=reg_callback_get_label),
            )
            for x in [train_folder, val_folder]

        }
    else:
        dataloaders = {
            test_folder: DataLoader(
                image_datasets[test_folder],
                batch_size=batch_size,
                num_workers=4,
                pin_memory=True
            )
        }


    dataset_sizes = {x: len(image_datasets[x]) for x in ([train_folder, val_folder] if istrain else [test_folder])}

    for x in ([train_folder, val_folder] if istrain else [test_folder]):
        print("Loaded {} images under {}".format(dataset_sizes[x], x))


    return dataloaders, dataset_sizes

def reg_eval_model(dataloaders, dataset_sizes, test_folder, model, criterion, device, is_gist=False, is_saliency=False):
    since = time.time()
    loss_test = 0
    labels_list = []
    preds_list = []

    # test_batches = len(dataloaders[test_folder])
    print("Evaluating model")
    print('-' * 10)

    for i, data in enumerate(dataloaders[test_folder]):
        # if i % 100 == 0:
        #     print("\rTest batch {}/{} \n".format(i, test_batches), end='', flush=True)
            #print(i, data['image'].size(), data['lable'].size())


        _, inputs, labels = data


        if not is_gist and not is_saliency:
            inputs = inputs.to(device)
        elif is_gist and is_saliency:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)
            inputs[2] = inputs[2].to(device)
        elif is_gist or is_saliency:
            inputs[0] = inputs[0].to(device)
            inputs[1] = inputs[1].to(device)

        labels = labels.to(device)

        model.train(False)
        model.eval()
        outputs = model(inputs)
        outputs = 1 + outputs * (9-1) #normalize to 1-9   Because our network last layer is sigmoid, which gives 0-1 values to restrict the range of outputs to be [0-1] or [1-9]
        loss = criterion(outputs.view(labels.size()), labels.float())

        loss_test += loss.data

        # print(i+1, '{:.4f}'.format(loss.cpu().item()))

        #here taking lots of time to fix the GPU out of Memory issue: need to  be numpy()
        labels_list.extend(labels.cpu().numpy())
        preds = outputs.reshape(labels.shape).cpu().detach().numpy()
        preds_list.extend(preds)

        #clean the cache
        del inputs, labels, outputs, loss, preds
        torch.cuda.empty_cache()

    # calculate correalation R
    avg_loss = loss_test / dataset_sizes[test_folder]
    score_c_r, _ = pearsonr(labels_list, preds_list)
    score_cv_r2 = np.around(score_c_r ** 2,2) #r2_score(labels_list, preds_list)

    # mse_loss = mean_squared_error(labels_list, preds_list)

    elapsed_time = time.time() - since
    # print()
    print("Evaluation completed in {:.0f}m {:.0f}s".format(elapsed_time // 60, elapsed_time % 60))
    print("Avg MSE loss (test): {:.4f}".format(avg_loss))
    print("R (test): {:.4f}".format(score_c_r))
    print('R2 : %5.4f' % score_cv_r2)
    # print('MSE CV: %5.4f' % mse_loss)

    #plot correatlion

    plot_correaltion(labels_list, preds_list, score_c_r, score_cv_r2, avg_loss,test_folder)

    print('-' * 10)
    return score_c_r, avg_loss