var documenterSearchIndex = {"docs":
[{"location":"trainer.html#Trainer","page":"Trainer","title":"Trainer","text":"","category":"section"},{"location":"trainer.html","page":"Trainer","title":"Trainer","text":"Trainer\nTsunami.fit!","category":"page"},{"location":"trainer.html#Tsunami.Trainer","page":"Trainer","title":"Tsunami.Trainer","text":"Trainer(; kws...)\n\nA type storing the training options to be passed to fit!.\n\nArguments\n\naccelerator: Supports passing different accelerator types (:cpu, :gpu,  :auto).               :auto will automatically select a gpu if available.               See also the devices option.                Default: :auto.\ncheckpointer: If true, enable checkpointing.                   Default: true.\ndefault_root_dir : Default path for logs and weights.                     Default: pwd().\ndevices: Pass an integer n to train on n devices,            or a list of devices ids to train on specific devices.           If nothing, will use all available devices.            Default: nothing.\nfast_dev_run: If set to true runs a single batch for train and validation to find any bugs.             Default: false.\nlogger: If true use tensorboard for logging.           Every output of the training_step will be logged every 50 steps.           Default: true.\nmax_epochs: Stop training once this number of epochs is reached.                Disabled by default (nothing).                If both max_epochs and max_steps are not specified,                defaults to max_epochs = 1000. To enable infinite training, set max_epochs = -1.               Default: nothing.\nmax_steps: Stop training after this number of steps.               Disabled by default (-1).               If max_steps = -1 and max_epochs = nothing, will default to max_epochs = 1000.               To enable infinite training, set max_epochs to -1.              Default: -1.\nprogress_bar: It true, shows a progress bar during training.                  Default: true.\nval_every_n_epoch: Perform a validation loop every after every N training epochs.                        Default: 1.\n\nExamples\n\ntrainer = Trainer(max_epochs = 10, \n                  accelerator = :cpu,\n                  checkpointer = true,\n                  logger = true)\n\nTsunami.fit!(model, trainer; train_dataloader, val_dataloader)\n\n\n\n\n\n","category":"type"},{"location":"trainer.html#Tsunami.fit!","page":"Trainer","title":"Tsunami.fit!","text":"fit!(model::FluxModule, trainer::Trainer; train_dataloader, val_dataloader = nothing, ckpt_path = nothing)\n\nTrain a model using the Trainer configuration. If ckpt_path is not nothing, training is resumed from the checkpoint.\n\nArguments\n\nmodel: A Flux model subtyping FluxModule.\ntrainer: A Trainer object storing the configuration options for fit!.\ntrain_dataloader: A DataLoader used for training. Required dargument.\nval_dataloader: A DataLoader used for validation. Default: nothing.\nckpt_path: Path of the checkpoint from which training is resumed. Default: nothing.\n\nExamples\n\ntrainer = Trainer(max_epochs = 10)\nTsunami.fit!(model, trainer; train_dataloader, val_dataloader)\n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#FluxModule","page":"FluxModule","title":"FluxModule","text":"","category":"section"},{"location":"fluxmodule.html","page":"FluxModule","title":"FluxModule","text":"FluxModule\nTsunami.configure_optimisers\nTsunami.test_step\nTsunami.test_epoch_end\nTsunami.training_step\nTsunami.training_epoch_end\nTsunami.validation_step\nTsunami.validation_epoch_end","category":"page"},{"location":"fluxmodule.html#Tsunami.FluxModule","page":"FluxModule","title":"Tsunami.FluxModule","text":"FluxModule\n\nAn abstract type for Flux models. A FluxModule helps orgainising you code and provides a standard interface for training.\n\nA FluxModule comes with functor already implemented. You can change the trainables by implementing Optimisers.trainables.\n\nTypes inheriting from FluxModule have to be mutable. They also have to implement the following methods in order to interact with a Trainer.\n\nRequired methods\n\nconfigure_optimisers(model)\ntraining_step(model, batch, batch_idx)\n\nOptional Methods\n\nvalidation_step(model, batch, batch_idx)\ntest_step(model, batch, batch_idx)\ntraining_epoch_end(model, outs)\nvalidation_epoch_end(model, outs)\ntest_epoch_end(model, outs)\n\nExamples\n\nusing Flux, Tsunami, Optimisers\n\n# Define a Multilayer Perceptron implementing the FluxModule interface\n\nmutable struct MLP <: FluxModule\n    net\nend\n\nfunction MLP()\n    net = Chain(Dense(4 => 32, relu), Dense(32 => 2))\n    return MLP(net)\nend\n\n(model::MLP)(x) = model.net(x)\n\nfunction Tsunami.training_step(model::MLP, batch, batch_idx)\n    x, y = batch\n    y_hat = model(x)\n    loss = Flux.Losses.mse(y_hat, y)\n    return loss\nend\n\nfunction Tsunami.configure_optimisers(model::MLP)\n    return Optimisers.setup(Optimisers.Adam(1e-3), model)\nend\n\n# Prepare the dataset and the DataLoader\n\nX, Y = rand(4, 100), rand(2, 100)\ntrain_dataloader = Flux.DataLoader((x, y), batchsize=10)\n\n\n# Create and Train the model\n\nmodel = MLP()\ntrainer = Trainer(max_epochs=10)\nTsunami.fit!(model, trainer; train_dataloader)\n\n\n\n\n\n","category":"type"},{"location":"fluxmodule.html#Tsunami.configure_optimisers","page":"FluxModule","title":"Tsunami.configure_optimisers","text":"configure_optimisers(model)\n\nReturn an optimisers' state initialazed for the model.\n\nExamples\n\nusing Optimisers\n\nfunction configure_optimisers(model::MyFluxModule)\n    return Optimisers.setup(AdamW(1e-3), model)\nend\n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#Tsunami.test_step","page":"FluxModule","title":"Tsunami.test_step","text":"test_step(model, batch, batch_idx)\n\nIf not implemented, the default is to use validation_step.\n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#Tsunami.test_epoch_end","page":"FluxModule","title":"Tsunami.test_epoch_end","text":"test_epoch_end(model::MyModule, outs)\n\nIf not implemented, the default is to use validation_epoch_end.\n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#Tsunami.training_step","page":"FluxModule","title":"Tsunami.training_step","text":"training_step(model, batch, batch_idx)\n\nShould return either a scalar loss or a NamedTuple with a scalar 'loss' field.\n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#Tsunami.training_epoch_end","page":"FluxModule","title":"Tsunami.training_epoch_end","text":"training_epoch_end(model, outs)\n\nIf not implemented, do nothing. \n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#Tsunami.validation_step","page":"FluxModule","title":"Tsunami.validation_step","text":"validation_step(model, batch, batch_idx)\n\nIf not implemented, the default is to use training_step. The return type has to be a NamedTuple.\n\n\n\n\n\n","category":"function"},{"location":"fluxmodule.html#Tsunami.validation_epoch_end","page":"FluxModule","title":"Tsunami.validation_epoch_end","text":"validation_epoch_end(model::MyModule, outs)\n\nIf not implemented, the default is to compute the mean of the  scalar outputs of validation_step.\n\n\n\n\n\n","category":"function"},{"location":"checkpoints.html#Checkpoints","page":"Checkpoints","title":"Checkpoints","text":"","category":"section"},{"location":"checkpoints.html","page":"Checkpoints","title":"Checkpoints","text":"Tsunami.Checkpointer\nTsunami.load_checkpoint","category":"page"},{"location":"checkpoints.html#Tsunami.Checkpointer","page":"Checkpoints","title":"Tsunami.Checkpointer","text":"Checkpointer(folder)\n\nSaves a FluxModule to folder after every training epoch.\n\n\n\n\n\n","category":"type"},{"location":"checkpoints.html#Tsunami.load_checkpoint","page":"Checkpoints","title":"Tsunami.load_checkpoint","text":"load_checkpoint(path)\n\nLoads a checkpoint that was saved to path.  Returns a namedtuple of the model and the optimizer.\n\nSee also: Checkpointer.\n\n\n\n\n\n","category":"function"},{"location":"index.html#Tsunami.jl","page":"Home","title":"Tsunami.jl","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"A high-level deep learning framework for the Julia language  that helps you focus and organize the relevant part of your code while removing the boilerplate. ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Tsunami  is built on top of Flux.jl and it is heavily inspired by pytorch-lightning.","category":"page"},{"location":"index.html#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Tsunami is still in an early development change and it is not a registered package yet.  Things can break without any notice. ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"Install Tsunami with ","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"pkg> add https://github.com/CarloLucibello/Tsunami.jl","category":"page"},{"location":"index.html#Features","page":"Home","title":"Features","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Use fit! instead of implementing a training loop.\nLogging (tensorboard).\nCheckpoints (save and resume training).\nGPU movement.","category":"page"},{"location":"index.html#Usage-Examples","page":"Home","title":"Usage Examples","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"Define your model subtyping the FluxModule abstract type, implement a few required methods, then let the Trainer train the model on your dataset with fit!. Tsunami will handle all of the boilerplate (training loop, loggin, gpu movement, validation, ...).","category":"page"},{"location":"index.html","page":"Home","title":"Home","text":"See the folder examples/ for usage examples.","category":"page"},{"location":"index.html#Similar-libraries","page":"Home","title":"Similar libraries","text":"","category":"section"},{"location":"index.html","page":"Home","title":"Home","text":"FastAI.jl\nFluxTraining.jl","category":"page"}]
}