function dir_str = UserInput_SSL_Demo(install_dir)
%%  Prompt user directions from the MATLAB command line for the SSL demo.
% The user can recreate Figures 3 and 4 or process their own data.

dir_str = []; % Initialization of directory assignement.
disp(['*** MATLAB is executing in ' install_dir ' ***'])

prompt = 'Do you want to reproduce a figure from the paper? [y/n]\n';
str = input(prompt,'s');
if isempty(str)
    str = 'y';
end

if strcmp(str,'y') %  "Reproduce a figure" branch.
    prompt = 'What figure number? [3/4]\n';
    str = input(prompt,'s');
    if strcmp(str,'3') % Selected Fig3.
        prompt = 'Which sub-figure? [a/b/c/d/e/f]\n';
        str = input(prompt,'s');
        if strcmp(str,'a')
            dir_str = [ install_dir '\fig3\a\data' ];
        elseif strcmp(str,'b')
            dir_str = [ install_dir '\fig3\b\data' ];
        elseif strcmp(str,'c')
            dir_str = [ install_dir '\fig3\c\data' ];
        elseif strcmp(str,'d')
            dir_str = [ install_dir '\fig3\d\data' ];
        elseif strcmp(str,'e')
            dir_str = [ install_dir '\fig3\e\data' ];
        elseif strcmp(str,'f')
            dir_str = [ install_dir '\fig3\f\data' ];
        end
    elseif strcmp(str,'4') % Selected Fig4.
        dir_str = [ install_dir '\fig4\data' ];
        disp('Processing will take approximately 10 minutes.')
    else
        disp('Not a valid choice');
    end
else % "Process user data" branch.
    prompt = 'OK, do you want to analyze your own data? [y/n]\n';
    str = input(prompt,'s');
    if strcmp(str,'n')
        disp('Mmmm ... not sure what you want.')
        dir_str = [];
    else % Are files in the directory we created for the user?
        prompt = 'OK, are the data files in "../UserData/data" ? [y/n]\n';
        str = input(prompt,'s');
        if strcmp(str,'y')
            dir_str = [ install_dir '\UserData\data' ];
        else
            disp('Please place your tiff files in the supplied "../UserData/data" folder.')
        end
    end
end

end


















