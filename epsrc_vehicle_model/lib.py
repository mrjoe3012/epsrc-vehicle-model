import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import scipy.stats as stats
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Tuple, Any
from pathlib import Path
from eufs_msgs.msg import CarState
from ugrdv_msgs.msg import VCUStatus, DriveRequest
from rclpy.serialization import deserialize_message
from scipy.spatial.transform import Rotation
from torch import TensorType
from tqdm import tqdm as tqdm
from numpy.typing import NDArray
import torch, sqlite3, os, pickle, json

class InputConstraints:
    SIZE = 7
    def __init__(self, means: TensorType, stds: TensorType):
        """
        Input and output distributions for normalisation.
        """
        self.means = means
        self.stds = stds

class OutputConstraints:
    SIZE = 8
    def __init__(self, means: TensorType, stds: TensorType):
        """
        Input and output distributions for normalisation.
        """
        self.means = means
        self.stds = stds

class OnlineStatistics:
    def __init__(self, size: int, dtype=np.float32):
        """
        Encapsulation of algorithms to calculate mean and standard deviations online.
        :param size: The number of elements in the data vector.
        :param dtype: The datatype of the elements within the data vector.
        """
        self._size = size
        self._dtype = dtype
        self._mean = np.zeros((self._size,), dtype=dtype)
        self._variance = np.zeros((self._size,), dtype=dtype)
        self._M = np.zeros((self._size,), dtype=self._dtype)
        self._n = 0
    def __call__(self, x: (NDArray | None) = None) -> Tuple[NDArray, NDArray]:
        """
        Updates internal means and standard deviations. Returns new values.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        :param x: The new data to ingest.
        :returns: The new mean followed by the new variance.
        """
        if x is not None:
            self._n += 1
            prev_diff = x - self._mean
            self._mean = self._mean + prev_diff / self._n
            curr_diff = x - self._mean
            self._M = self._M + np.multiply(prev_diff, curr_diff)
            if self._n > 1: self._variance = self._M / (self._n - 1)
        return self._mean, self._variance

class DataCollater:
    def __init__(self, car_states: List[CarState], vcu_statuses: List[VCUStatus], drive_requests: List[DriveRequest]):
        """
        Organises different messages, grouping them by timestamp.
        :param car_states: CarState messages
        :param vcu_statuses: VCUStatus messages
        :param drive_requests: DriveRequest messages
        """
        self._car_states = car_states
        self._vcu_statuses = vcu_statuses
        self._drive_requests = drive_requests
    
    ####################
    # public interface #
    ####################

    def collate(self, nearby_threshold: float, max_delta_time: float) \
            -> Tuple[list, list]:
        """
        Transforms lists of ROS messages into x, y tensors. Does this by associating messages
        based on timestamp and calculating deltas between these consecutive blocks.
        :param nearby_threshold: Maximum milliseconds between messages for them to be grouped together.
        :param max_delta_time: Any data rows with time delta (seconds) greather than this value are filtered
        out of the result.
        :returns: x,y tensors for training.
        """ 
        entries = []
        # condense individual messages into entries
        self._condense_messages(self._car_states, "car_state", nearby_threshold, entries)
        self._condense_messages(self._vcu_statuses, "vcu_status", nearby_threshold, entries)
        self._condense_messages(self._drive_requests, "drive_request", nearby_threshold, entries)
        # remove entries without all messages
        entries = [
            entry for entry in entries if entry['vcu_status'] and entry['car_state'] and entry['drive_request'] \
        ]
        entries.sort(key=lambda entry: entry['timestamp'])
        # now convert list of entries into tensor form
        xs = []
        ys = []
        for i in range(len(entries) - 1):
            curr = entries[i]
            next = entries[i+1]
            dt = (next['timestamp'] - curr['timestamp']) / 1000
            if dt > max_delta_time: continue  # don't use data if time jump is too large
            x, y = self._make_data_rows_xy(curr, next)
            xs.append(x)
            ys.append(y)
            
        return xs, ys

    ###################
    # private methods #
    ###################

    @staticmethod
    def _make_data_rows_xy(curr: dict, next: dict) -> Tuple[List[float], List[float]]:
        """
        Convert consecutive entries into a data row.
        """
        # x: [steering_angle, wheel_rpms, steering_angle_request, acceleration_request]
        # y: d[x, y, theta, steering_angle, wheel_rpms] / dt
        dt = (next['timestamp'] - curr['timestamp']) / 1000
        x = [
            curr['vcu_status'].steering_angle,
            curr['vcu_status'].wheel_speeds.fl_speed,
            curr['vcu_status'].wheel_speeds.fr_speed,
            curr['vcu_status'].wheel_speeds.rl_speed,
            curr['vcu_status'].wheel_speeds.rr_speed,
            curr['drive_request'].ackermann.drive.steering_angle,
            curr['drive_request'].ackermann.drive.acceleration
        ]
        # global position delta
        dx_g = next['car_state'].pose.pose.position.x - curr['car_state'].pose.pose.position.x
        dy_g = next['car_state'].pose.pose.position.y - curr['car_state'].pose.pose.position.y
        # local deltas
        theta = DataCollater._get_heading(curr['car_state'])
        dx = dx_g * np.cos(-theta) - dy_g * np.sin(-theta)
        dy = dx_g * np.sin(-theta) + dy_g * np.cos(-theta) 
        curr_heading_quat = Rotation.from_quat([
            curr['car_state'].pose.pose.orientation.x,
            curr['car_state'].pose.pose.orientation.y,
            curr['car_state'].pose.pose.orientation.z,
            curr['car_state'].pose.pose.orientation.w,
        ])
        next_heading_quat = Rotation.from_quat([
            next['car_state'].pose.pose.orientation.x, 
            next['car_state'].pose.pose.orientation.y, 
            next['car_state'].pose.pose.orientation.z, 
            next['car_state'].pose.pose.orientation.w, 
        ])
        dtheta = (next_heading_quat * curr_heading_quat.inv()).as_euler("XYZ")[2]
        dsteer = next['vcu_status'].steering_angle - curr['vcu_status'].steering_angle
        dflspeed = next['vcu_status'].wheel_speeds.fl_speed - curr['vcu_status'].wheel_speeds.fl_speed
        dfrspeed = next['vcu_status'].wheel_speeds.fr_speed - curr['vcu_status'].wheel_speeds.fr_speed
        drlspeed = next['vcu_status'].wheel_speeds.rl_speed - curr['vcu_status'].wheel_speeds.rl_speed
        drrspeed = next['vcu_status'].wheel_speeds.rr_speed - curr['vcu_status'].wheel_speeds.rr_speed
        y = [
            dx / dt,
            dy / dt,
            dtheta / dt,
            dsteer / dt,
            dflspeed / dt,
            dfrspeed / dt,
            drlspeed / dt,
            drrspeed / dt
        ]
        return x, y

    @staticmethod
    def _condense_messages(messages: list, id: str, nearby_threshold: float, entries: list) -> None:
        """
        Takes a list of separate messages and attempts to group them by
        timestamp.
        """
        for (t, msg) in messages:
            nearest = DataCollater._nearest_entry(entries, t, nearby_threshold)
            if nearest is None:
                entries.append(DataCollater._make_entry(t, **{id : msg}))
            else:
                nearest[id] = msg

    @staticmethod
    def _make_entry(timestamp: float, car_state=None, vcu_status=None, drive_request=None) -> dict:
        """
        Represent a group of close-together-in-time messages.
        """
        return {
            'timestamp' : timestamp,
            'car_state' : car_state,
            'vcu_status' : vcu_status,
            'drive_request' : drive_request
        }
    
    @staticmethod
    def _nearest_entry(entries: List[dict], timestamp: float, threshold: float) -> dict | None:
        """
        Finds the entry in entries with the timestamp closest to timestamp,
        or returns None.
        """
        nearest = None
        nearest_delta = np.inf
        for entry in entries:
            delta = abs(entry['timestamp'] - timestamp)
            if delta < nearest_delta:
                nearest_delta = delta
                nearest = entry
        if nearest is not None and nearest_delta > threshold:
            nearest = None
        return nearest 

    @staticmethod
    def _get_heading(car_state: CarState) -> float:
        """
        Get a CarState's heading
        """
        quat = Rotation.from_quat([
            car_state.pose.pose.orientation.x,
            car_state.pose.pose.orientation.y,
            car_state.pose.pose.orientation.z,
            car_state.pose.pose.orientation.w,
        ])
        return quat.as_euler("XYZ")[2]

class SimDataStream:
    def __init__(self, path: str | Path):
        """
        Stream-like interface for simulation data.
        :param path: The dataset binary file.
        """
        super().__init__()
        path = Path(path)
        self._f = open(path, "rb")

    def __iter__(self):
        return self

    def __next__(self):
        return self._read_next()

    def __del__(self):
        self._f.close()

    def _read_next(self) -> Tuple[TensorType, TensorType]:
        """
        Deserialization of binary simulation data stored in blocks with
        the following structure:
        ----start of block------
        length of x tensor in bytes (2 byte integer big endian)
        x tensor (serialized numpy array)
        length of y tensor in bytes (2 byte integer big endian)
        y tensor (serialized numpy array)
        ----end of block-------
        """
        xsize = self._f.read(2)
        if len(xsize) != 2: raise StopIteration()
        xsize = int.from_bytes(xsize, "big")
        xbytes = self._f.read(xsize)
        if len(xbytes) != xsize: raise RuntimeError()
        ysize = self._f.read(2)
        if len(ysize) != 2: raise RuntimeError()
        ysize = int.from_bytes(ysize, "big")
        ybytes = self._f.read(ysize)
        if len(ybytes) != ysize: raise RuntimeError()
        x = np.frombuffer(xbytes, dtype=np.float32)
        y = np.frombuffer(ybytes, dtype=np.float32)
        return torch.tensor(x), torch.tensor(y) 

class SimData(Dataset):
    def __init__(self, path: str | Path):
        """
        In-memory dataset for simulation data.
        :param path: The dataset binary file.
        """
        super().__init__()
        self._path = path
        self._cache = self._cache_data()

    def __len__(self):
        return len(self._cache)

    def __getitem__(self, idx):
        return self._cache[idx]

    def _cache_data(self) -> List[Tuple[TensorType, TensorType]]:
        """
        Use a stream object to read an entire dataset into a list.
        """
        stream = SimDataStream(self._path)
        result = [(x,y) for (x,y) in stream]
        return result

    ####################
    # public interface #
    ####################

    @staticmethod
    def preprocess_dataset(root_dirs: List[str | Path] | (str | Path), train_output: str | Path,
                           test_output: str | Path, constraints_output: str | Path,
                           train: float, verbose: bool = False) -> None:
        """
        Converts from raw simulation data into a compact binary training/testing data format.
        :param root_dirs: A single root directory containing simulation databases, or a list of them.
        :param train_output: Filename for the training dataset.
        :param test_output: Filename for the testing dataset.
        :param constraints_output: Filename for dataset constraints output. Will be a pickled Tuple[InputConstraints, OutputConstraints]
        :param train: Proportion of data to use for training (0,1]. If 1, test dataset will be empty.
        :param verbose: Print info and progress bars.
        """
        assert 0.0 <= train <= 1.0
        if type(root_dirs) != list:
            root_dirs = [root_dirs]
        root_dirs = [Path(x) for x in root_dirs]
        # flat array of database paths
        db_paths = [
            root_dir / x for root_dir in root_dirs for x in os.listdir(root_dir)
        ]
        constraints_output = Path(constraints_output)
        if verbose == True:
            print(f"Processing a total of {len(db_paths)} databases.")
        if train < 1.0:
            train_idx = int(len(db_paths) * train)
            db_paths_train = db_paths[:train_idx]
            db_paths_test = db_paths[train_idx:]
            (ic1, oc1), (ic2, oc2) = SimData._preprocess_databases(db_paths_train, train_output, verbose=verbose), \
                SimData._preprocess_databases(db_paths_test, test_output, verbose=verbose)
            with open(constraints_output, "wb") as f:
                pickle.dump((ic1, oc1), open(constraints_output, "wb"))
            return (ic1, oc1), (ic2, oc2)
        else:
            (ic1, oc1), (ic2, oc2) = SimData._preprocess_databases(db_paths, train_output, verbose=verbose), \
                SimData._preprocess_databases([], test_output, verbose=False)
            with open(constraints_output, "wb") as f:
                pickle.dump((ic1, oc1), f)
            return (ic1, oc1), (ic2, oc2)

    @staticmethod
    def plot_dataset(path: str | Path, constraints_path: str | Path,
                     show: bool = True, save_dir: (str | Path | None) = None) -> None:
        """
        Plots the dataset. Warning, this will attempt to put the entire dataset in memory.
        :param path: Path to the dataset binary.
        :param constraints_path: Path to the pickled dataset constraints.
        """
        output_names = ['longitudinal velocity', 'lateral velocity', 'yaw rate', 'steering angle rate', 'fl wheel speed', 'fr wheel speed', 'rl wheel speed', 'rr wheel speed']
        input_names = ['steering angle', 'fl wheel speed', 'fr wheel speed', 'rl wheel speed', 'rr wheel speed', 'steering angle request', 'acceleration request']
        path = Path(path)
        constraints_path = Path(constraints_path)
        dataset = SimData(path, in_memory=True)
        if save_dir is not None:
            save_dir = Path(save_dir)
        with open(constraints_path, "rb") as f:
            (input_constraints, output_constraints) = pickle.load(f)
        input_data = np.array([
            x.cpu().tolist() for x, _ in dataset._cache
        ])
        output_data = np.array([
            y.cpu().tolist() for _, y in dataset._cache
        ])
        size = (12, 9)
        hist_bins = 20
        # histograms for each element
        # with normal distribution fit
        for i in range(input_constraints.SIZE):
            fig = plt.figure(figsize=size)
            plt.title(f"input: {input_names[i]}")
            plt.hist(input_data[:, i], bins=hist_bins, density=True)
            mean, std = input_constraints.means[i].item(), input_constraints.stds[i].item()
            norm_x = np.linspace(mean - 3*std, mean + 3*std)
            plt.plot(
                norm_x,
                stats.norm.pdf(norm_x, mean, std)
            )
            if save_dir is not None:
                plt.savefig(save_dir / f"input-{input_names[i]}.png".replace(" ", "-"))
            if show == True:
                plt.show()
            plt.close(fig)
            plt.cla()
            plt.clf()
        for i in range(output_constraints.SIZE):
            fig = plt.figure(figsize=size)
            plt.title(f"output: {output_names[i]}")
            plt.hist(output_data[:, i], bins=hist_bins, density=True)
            mean, std = output_constraints.means[i].item(), output_constraints.stds[i].item()
            norm_x = np.linspace(mean - 3*std, mean + 3*std)
            plt.plot(
                norm_x,
                stats.norm.pdf(norm_x, mean, std)
            )
            if save_dir is not None:
                plt.savefig(save_dir / f"output-{output_names[i]}.png".replace(" ", "-"))
            if show == True:
                plt.show()
            plt.close(fig)
            plt.cla()
            plt.clf()

    ###################
    # private methods #
    ###################

    @staticmethod
    def _preprocess_databases(db_paths: List[Path], output: Path, verbose: bool = False) -> Tuple[InputConstraints, OutputConstraints]:
        x_stats = OnlineStatistics(InputConstraints.SIZE, np.float32)
        y_stats = OnlineStatistics(OutputConstraints.SIZE, np.float32)
        k = 0
        with open(output, "wb") as f:
            for db_path in tqdm(db_paths, "Preprocessing", disable=not verbose):
                car_states, vcu_statuses, drive_requests = SimData._read_database(db_path)
                xs, ys = SimData._convert_messages_to_lists(car_states, vcu_statuses, drive_requests)
                assert len(xs) == len(ys)
                for i in range(len(xs)):
                    x = np.array(xs[i], dtype=np.float32)
                    y = np.array(ys[i], dtype=np.float32)
                    x_stats(x)
                    y_stats(y)
                    x_bin = x.tobytes()
                    y_bin = y.tobytes()
                    x_size = len(x_bin).to_bytes(length=2, byteorder="big")
                    y_size = len(y_bin).to_bytes(length=2, byteorder="big")
                    f.write(x_size + x_bin + y_size + y_bin)
            mean_xs, var_xs = x_stats()
            mean_ys, var_ys = y_stats()
            std_xs, std_ys = np.sqrt(var_xs), np.sqrt(var_ys)
            std_xs[std_xs == 0.0] = 1.0  # any std too small to calculate is set to 1
            std_ys[std_ys == 0.0] = 1.0
            print(f"Input (mean, std): {np.vstack([mean_xs, std_xs]).T}")
            print(f"Output (mean, std): {np.vstack([mean_ys, std_ys]).T}")
            return InputConstraints(torch.tensor(mean_xs), torch.tensor(std_xs)), \
                OutputConstraints(torch.tensor(mean_ys), torch.tensor(std_ys))


    @staticmethod
    def _read_database(db_path: Path) -> Tuple[List[CarState], List[VCUStatus], List[DriveRequest]]:
        """
        Reads an sqlite3 database, extracting the VCUStatus, CarState and
        DriveRequest messages and deserializing them.
        :param db_path: The path to the database to read.
        :returns: Lists of VCUStatus, DriveRequests and CarStates
        """
        connection = sqlite3.connect(db_path)
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT timestamp, data FROM ground_truth_state")
            car_states_bin = cursor.fetchall()
            cursor.execute("SELECT timestamp, data FROM vcu_status")
            vcu_statuses_bin = cursor.fetchall()
            cursor.execute("SELECT timestamp, data FROM drive_request")
            drive_requests_bin = cursor.fetchall()
            car_states = [
                (t, deserialize_message(cs_bin, CarState)) for (t,cs_bin) in car_states_bin
            ]
            vcu_statuses = [
                (t, deserialize_message(vcu_bin, VCUStatus)) for (t,vcu_bin) in vcu_statuses_bin
            ]
            drive_requests = [
                (t, deserialize_message(drive_bin, DriveRequest)) for (t,drive_bin) in drive_requests_bin
            ]
        finally:
            connection.close()
        return car_states, vcu_statuses, drive_requests

    @staticmethod
    def _convert_messages_to_lists(car_states: List[CarState],
                                     vcu_statuses: List[VCUStatus],
                                     drive_requests: List[DriveRequest]) \
                                     -> Tuple[list, list]:
        collater = DataCollater(car_states, vcu_statuses, drive_requests)
        max_delta_time = 0.4
        nearby_threshold = 5.0
        return collater.collate(nearby_threshold, max_delta_time)

class VehicleModel(nn.Module):
    def __init__(self, input_constaints: InputConstraints, output_constraints: OutputConstraints,
                 num_neurons: int = 32, num_layers: int = 2, verbose: bool = True):
        """
        Models a vehicle's dynamics using a neural network.
        :param input_constraints: Constants used for input normalisation.
        :param output_constraitns: Constants used for output normalisation.
        :param num_neurons: Number of neurons in each hidden layer.
        :param num_layers: Total number of layers.
        """
        super().__init__()
        assert num_layers >= 1
        self._input_constraints = input_constaints
        self._output_constraints = output_constraints
        self._verbose = verbose
        layers = []
        # input
        layers += [
            nn.Linear(InputConstraints.SIZE, num_neurons),
            nn.ReLU()
        ]
        # hidden
        for i in range(num_layers - 1):
            layers += [
                nn.Linear(num_neurons, num_neurons),
                nn.ReLU()
            ]
        # output
        layers += [
            nn.Linear(num_neurons, OutputConstraints.SIZE)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.normalise_input(x)
        y_pred = self.net(x)
        if self.training == False:
            y_pred = self.normalise_output(y_pred, denormalise=True)
        return y_pred

    def normalise_input(self, x: TensorType, denormalise: bool = False) -> TensorType: 
        if denormalise == False:
            return (x - self._input_constraints.means) / self._input_constraints.stds
        else:
            return (x * self._input_constraints.stds) + self._input_constraints.means

    def normalise_output(self, y: TensorType, denormalise: bool = False) -> TensorType:
        if denormalise == False:
            return (y - self._output_constraints.means) / self._output_constraints.stds
        else:
            return (y * self._output_constraints.stds) + self._output_constraints.means

    def train_loop(self, dataset: SimData, batch_size: int, epochs: int, optimizer: torch.optim.Optimizer = None) -> None:
        self.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cuda:0"))
        if optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())
        loss_fn = nn.MSELoss()
        errors = []
        for i in range(epochs):
            loss_sum = 0.0
            for batch, (x, y) in enumerate(dataloader):
                optimizer.zero_grad()
                y_pred = self(x)
                loss = loss_fn(y_pred, self.normalise_output(y))
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
            loss_sum /= (batch + 1)
            loss_sum = loss_sum**0.5  # RMSE
            errors.append(loss_sum)
            if (i + 1) % 100 == 0 and self._verbose:
                print(f"Epoch #{i+1}: RMSE: {loss_sum}")
        return errors 

    def test_loop(self, dataset: SimData, batch_size: int) -> None:
        self.train()
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cuda:0"))
        with torch.no_grad():
            loss_fn = nn.MSELoss()
            loss_sum = 0.0
            for batch, (x, y) in enumerate(dataloader):
                y_pred = self(x)
                loss_sum += loss_fn(y_pred, self.normalise_output(y)).item()
            loss_sum /= batch + 1
            loss_sum = loss_sum**0.5  # RMSE
            if self._verbose:
                print(f"Test RMSE: {loss_sum}")
            return loss_sum

    def plot_predictions(self, dataset: SimData, output_directory: Path | None = None, show: bool = True):
        size = (12.8, 9.6)
        plt.cla()
        plt.clf()
        output_names = ['longitudinal velocity', 'lateral velocity', 'yaw rate', 'steering angle rate', 'fl wheel speed', 'fr wheel speed', 'rl wheel speed', 'rr wheel speed']
        input_names = ['steering angle', 'fl wheel speed', 'fr wheel speed', 'rl wheel speed', 'rr wheel speed', 'steering angle request', 'acceleration request']
        self.train()
        batch_size = 4096
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator(device="cuda:0"))
        with torch.no_grad():
            losses = torch.zeros((self._output_constraints.SIZE), dtype=torch.float32)
            data_x = None
            data_y = None
            data_y_pred = None
            for batch, (x, y) in enumerate(dataloader):
                y_pred = self(x)
                loss = torch.sum(torch.square(y_pred - self.normalise_output(y)), dim=0)
                losses = losses + loss
                if data_x is None:
                    data_x = self.normalise_input(x)
                    data_y = self.normalise_output(y)
                    data_y_pred = y_pred
                else:
                    data_x = torch.vstack((data_x, self.normalise_input(x)))
                    data_y = torch.vstack((data_y, self.normalise_output(y)))
                    data_y_pred = torch.vstack((data_y_pred, y_pred))
            data_x = data_x.T.cpu().numpy()
            data_y = data_y.T.cpu().numpy()
            data_y_pred = data_y_pred.T.cpu().numpy()
            losses = losses / (batch + 1)  # element-wise RMSE
            losses = losses.cpu().numpy()
            # plot histograms of input/output data
            hist_bins = 20
            for i in range(self._input_constraints.SIZE):
                fig = plt.figure(figsize=size)
                plt.title(f"Input: {input_names[i]}")
                plt.hist(data_x[i,:], bins=hist_bins)
                if output_directory:
                    plt.savefig(output_directory / f"input-{input_names[i]}.png".replace(" ", "-"))
                if show == True: plt.show()
                plt.close(fig)
                plt.cla()
                plt.clf()
            for i in range(self._output_constraints.SIZE):
                fig = plt.figure(figsize=size)
                plt.title(f"Output: {output_names[i]}")
                plt.hist(data_y[i, :], bins=hist_bins, color="blue", alpha=0.75, density=True)
                plt.hist(data_y_pred[i, :], bins=hist_bins, color="red", alpha=0.75, density=True)
                plt.legend(["Model Output", "Targets"])
                if output_directory:
                    plt.savefig(output_directory / f"output-{output_names[i]}.png".replace(" ", "-"))
                if show == True: plt.show()
                plt.close(fig)
                plt.cla()
                plt.clf()
            # plot scatterplots of predictions
            for i in range(self._output_constraints.SIZE):
                fig = plt.figure(figsize=size)
                plt.title(f"Predictions: {output_names[i]}")
                plt.plot(
                    data_y[i, :],
                    data_y_pred[i, :],
                    "o",
                    markersize=1.0,
                    alpha=0.4
                )
                plt.axis("equal")
                plt.xlabel("Target")
                plt.ylabel("Output")
                if output_directory:
                    plt.savefig(output_directory / f"predictions-{output_names[i]}.png".replace(" ", "-"))
                if show == True: plt.show()
                plt.close(fig)
                plt.cla()
                plt.clf() 