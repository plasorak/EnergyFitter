



void Dumper() {

  TFile *f = new TFile("gaushit.root", "READ");
  int ChanWidth,Type,NHits,Config,Event;
  float SumADC,TimeWidth;
  double ENu_Lep;

  
  TTree* tree = (TTree*)f->Get("t_Output");
  tree->SetBranchAddress("Config", &Config   );
  tree->SetBranchAddress("ChanWidth", &ChanWidth);
  tree->SetBranchAddress("Event", &Event);
  tree->SetBranchAddress("Type", &Type     );
  tree->SetBranchAddress("NHits", &NHits    );
  tree->SetBranchAddress("SumADC", &SumADC   );
  tree->SetBranchAddress("TimeWidth", &TimeWidth);
  tree->SetBranchAddress("ENu", &ENu_Lep  );

  ofstream myfile;
  for(int i=0; i<6; ++i) {
    std::cout << "Doing config " << i << std::endl;
    myfile.open (Form("Config%i.txt", i));
    myfile << "Event,ChanWidth,Type,NHits,SumADC,TimeWidth,ENu"<<std::endl;
    for(int j=0;j<tree.GetEntries();++j) {
      if(j%10000==0)
        std::cout << "did " << 100.*j/tree->GetEntries() << std::endl;
      tree->GetEntry(j);
      if(Config!=i)
        continue;
      myfile << Event<<","<<ChanWidth<<","<<Type<<","<<NHits<<","<<SumADC<<","<<TimeWidth<<","<<ENu_Lep*1000<<std::endl;
    }
    myfile.close();
  }
}


